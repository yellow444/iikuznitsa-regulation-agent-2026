from contextlib import closing
from hmac import compare_digest
import logging
import os
from pathlib import Path
import re
import sqlite3

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, StringConstraints
from typing import Annotated, Literal

from retrieval import Catalog, SOURCES, terms


ROOT = Path(__file__).resolve().parent
CORPUS = Path(os.environ.get("CORPUS_DIR", ROOT.parent / "private-corpus"))
DB = Path(os.environ.get("QUALITY_DB", ROOT / "runtime" / "quality.sqlite3"))
TOKEN = os.environ.get("QUALITY_TOKEN", "")
PUBLIC_DEMO = os.environ.get("PUBLIC_DEMO") == "1"
MODEL_DIR = Path(os.environ.get("E5_MODEL_DIR", ROOT / "runtime" / "e5"))
logger = logging.getLogger("uvicorn.error")


def load_catalog():
    if all((MODEL_DIR / name).is_file() for name in ("model.onnx", "tokenizer.json")):
        try:
            return Catalog(CORPUS, model_dir=MODEL_DIR)
        except Exception:
            logger.exception("Модель E5 не запустилась. Включён лексический поиск.")
    return Catalog(CORPUS)


catalog = load_catalog()
app = FastAPI(title="Регламент · поиск корпоративных знаний")
Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=4, max_length=1400)]
Answer = Annotated[str, StringConstraints(strip_whitespace=True, min_length=10, max_length=3000)]


class Inquiry(BaseModel):
    question: Question
    scope: str | None = None
    condition: Literal["Меняют", "Не меняют", "Да", "Нет", "Не знаю"] | None = None
    suggest_approved: bool = True


class Resolution(BaseModel):
    answer: Answer
    source: str
    reference: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=240)]


def connect():
    DB.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY,
            question TEXT NOT NULL,
            question_key TEXT NOT NULL,
            scope TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            answer TEXT,
            source TEXT,
            reference TEXT,
            resolved_at TEXT,
            UNIQUE(question_key, scope)
        )
    """)
    return db


def question_key(value: str) -> str:
    return " ".join(re.findall(r"[а-яa-z0-9]+", value.casefold().replace("ё", "е")))


def approved_result(row):
    source_urls = {label: f"https://drive.google.com/file/d/{file_id}/view"
                   for label, file_id in SOURCES.values()}
    return {"state": "approved", "url": source_urls.get(row["source"]),
            **{key: row[key] for key in ("id", "answer", "source", "reference", "resolved_at")}}


def approved_suggestions(db, question: str, scope: str):
    query = terms(question)
    if len(query) < 2:
        return []
    number_pattern = re.compile(r"\b\d+\b")
    query_numbers = set(number_pattern.findall(question))
    rows = db.execute(
        "SELECT question, source, scope FROM requests WHERE resolved_at IS NOT NULL AND scope = ? "
        "ORDER BY id DESC LIMIT 100", (scope,),
    )
    guarded = quality_guards(question)
    anchors = quality_anchors(question)
    matches = []
    semantic_candidates = []
    for row in rows:
        candidate = terms(row["question"])
        if (query_numbers != set(number_pattern.findall(row["question"]))
                or guarded != quality_guards(row["question"])):
            continue
        common = len(query & candidate)
        similarity = 2 * common / (len(query) + len(candidate))
        if common >= 2 and similarity >= 0.60:
            matches.append((similarity, dict(row)))
        elif catalog.encoder is not None and len(anchors & quality_anchors(row["question"])) >= 2:
            semantic_candidates.append(dict(row))
    if semantic_candidates:
        texts = [question, *(item["question"] for item in semantic_candidates)]
        vectors = catalog.encoder.encode(texts, "query")
        for score, item in zip(vectors[1:] @ vectors[0], semantic_candidates):
            if score >= 0.84:
                matches.append((float(score), item))
    matches.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in matches[:3]]


QUALITY_GUARDS = {
    "cancellation": r"отмен\w*|аннулир\w*|отзыв\w*|отозва\w*|откат\w*",
    "transfer": (r"перенос\w*|перенес\w*|перенест\w*|переоформ\w*|перерегистр\w*|"
                 r"перевод\w*|перевест\w*|перевед\w*|переда(?:ть|ча|чи|че|чу)"),
    "activate": r"активировать|активаци\w*",
    "nonactive": r"неактивир\w*|неактивн\w*|не\s+активир\w*",
    "negative": r"не|нельзя|без|запрещ\w*|отказ\w*",
    "deadline": r"срок\w*|дедлайн\w*",
    "payment": r"оплат\w*|платеж\w*",
}
QUALITY_ANCHORS = {
    "organization": r"юрлиц\w*|организаци\w*|компани\w*|юридическ\w*\s+лиц\w*",
    "license": r"лицензи\w*|ключ\w*",
    "contract": r"договор\w*|контракт\w*",
    "access": r"доступ\w*|разрешени\w*",
}


def quality_guards(value: str) -> set[str]:
    normalized = value.casefold().replace("ё", "е")
    return {name for name, pattern in QUALITY_GUARDS.items()
            if re.search(rf"\b(?:{pattern})\b", normalized)}


def quality_anchors(value: str) -> set[str]:
    normalized = value.casefold().replace("ё", "е")
    return terms(value) | {name for name, pattern in QUALITY_ANCHORS.items()
                           if re.search(rf"\b(?:{pattern})\b", normalized)}


def authorize(token: str) -> None:
    if not TOKEN or not compare_digest(TOKEN.encode(), token.encode()):
        raise HTTPException(403, "Требуется ключ доступа службы качества.")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(ROOT / "index.html")


@app.get("/api/catalog")
def get_catalog():
    return {"sources": list(catalog.documents.values()), "passages": len(catalog.passages),
            "search": "hybrid" if catalog.encoder else "lexical", "public_demo": PUBLIC_DEMO}


@app.post("/api/ask")
def ask(inquiry: Inquiry):
    if not catalog.passages:
        raise HTTPException(503, "Регламенты ещё не загружены.")
    scope = inquiry.scope or ""
    if scope and scope not in catalog.documents.values():
        raise HTTPException(422, "Выберите регламент из списка.")
    if not question_key(inquiry.question):
        raise HTTPException(422, "Введите вопрос словами.")
    ticket_question = inquiry.question
    if inquiry.condition:
        ticket_question += f" [Уточнение: {inquiry.condition}]"
    key = question_key(ticket_question)
    with closing(connect()) as db:
        row = db.execute(
            "SELECT * FROM requests WHERE question_key = ? AND scope = ? AND resolved_at IS NOT NULL",
            (key, scope),
        ).fetchone()
    if row:
        return approved_result(row)
    result = catalog.find(inquiry.question, scope, inquiry.condition)
    if result["state"] in {"answer", "clarify"}:
        return result
    with closing(connect()) as db, db:
        if inquiry.suggest_approved and not inquiry.condition:
            suggestions = approved_suggestions(db, inquiry.question, scope)
            if suggestions:
                return {"state": "confirm_approved", "suggestions": suggestions}
        db.execute(
            "INSERT INTO requests(question, question_key, scope) VALUES(?, ?, ?) "
            "ON CONFLICT(question_key, scope) DO NOTHING",
            (ticket_question, key, scope),
        )
        row = db.execute(
            "SELECT id, answer, source, reference, resolved_at FROM requests "
            "WHERE question_key = ? AND scope = ?",
            (key, scope),
        ).fetchone()
        if row["resolved_at"]:
            return approved_result(row)
        return {"state": "escalated", "id": row["id"]}


@app.get("/api/quality")
def list_requests(x_quality_token: Annotated[str, Header()] = ""):
    authorize(x_quality_token)
    with closing(connect()) as db:
        rows = db.execute(
            "SELECT id, question, scope, created_at, answer, source, reference, resolved_at "
            "FROM requests ORDER BY resolved_at IS NOT NULL, id DESC LIMIT 100"
        ).fetchall()
    return [dict(row) for row in rows]


@app.post("/api/quality/{request_id}")
def resolve(request_id: int, resolution: Resolution, x_quality_token: Annotated[str, Header()] = ""):
    authorize(x_quality_token)
    if resolution.source not in catalog.documents.values():
        raise HTTPException(422, "Укажите регламент из загруженного каталога.")
    with closing(connect()) as db, db:
        updated = db.execute(
            "UPDATE requests SET answer = ?, source = ?, reference = ?, resolved_at = CURRENT_TIMESTAMP "
            "WHERE id = ? AND resolved_at IS NULL AND (scope = '' OR scope = ?)",
            (resolution.answer, resolution.source, resolution.reference, request_id, resolution.source),
        )
        if not updated.rowcount:
            raise HTTPException(409, "Запрос отсутствует, закрыт или привязан к другому регламенту.")
    return {"state": "approved", "id": request_id}
