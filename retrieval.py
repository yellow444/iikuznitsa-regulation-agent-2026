from collections import Counter
from dataclasses import asdict, dataclass
from math import log
from pathlib import Path
import re

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


SOURCES = {
    "dbrct.docx": ("ДБРЦТ · проекты", "1wuGujAe3Tm5zsiGmhMTxx31QaIf8pMlD"),
    "ib-project.docx": ("Дирекция ИБ · проекты", "1gujKSVIlDCSJsXxBeGGZ3D0mHjXCCtvc"),
    "tech.docx": ("Техническая дирекция · проекты", "1pEDQaQbvB8b-O15U7c-CN6HdYNJjbKh6"),
    "ib-presales.docx": ("Дирекция ИБ · пресейл", "1UMdoQnkBk0MshefJVtS061r-Asyt49aL"),
}
STOP = set(("как кто когда где что это при для или если либо какой какая какие "
            "кто такой куда чем чтобы должен нужно можно является будет "
            "когда кому кого что делать происходит поступить сотрудник подскажите пожалуйста").split())
ENDING = re.compile(r"(иями|ями|ами|ого|ему|ому|ыми|ими|ение|ания|ения|аний|"
                    r"ению|ает|яет|ить|ать|ять|ует|уют|ова|ией|иям|ия|ию|ие|"
                    r"ый|ой|ий|ая|ое|ые|ых|их|ов|ам|ах|ей|ом|ем|ть|[аеиоуюыяь])$")
ABBREVIATIONS = {"рп", "па", "ам", "рм", "ап", "тз", "иб", "ss"}
COMPOUND = re.compile(
    r"(?:\?\s*|,?\s+и\s+)(?=(?:кто|как|куда|когда|кому|какой|какая|какие|сколько|где|что|можно\s+ли)\b)",
    re.IGNORECASE,
)
LIST_REQUEST = re.compile(r"\b(?:какие|перечень|список|состав|комплект|приложить)\b", re.IGNORECASE)


def terms(value: str) -> set[str]:
    words = re.findall(r"[а-яa-z0-9]+", value.casefold().replace("ё", "е"))
    result = set()
    for word in words:
        if word in STOP or (len(word) < 3 and word not in ABBREVIATIONS):
            continue
        if word.startswith(("назнач", "поруч")):
            result.add("назнач")
            continue
        if word.startswith(("выбир", "выбор", "выбрат")):
            result.add("выбор")
            continue
        stem = ENDING.sub("", word) if len(word) > 4 and word.isalpha() else word
        result.add(stem if len(stem) >= 3 else word)
    return result


@dataclass(frozen=True)
class Passage:
    source: str
    section: str
    quote: str
    url: str


def read_passages(path: Path, source: str, file_id: str) -> tuple[list[Passage], dict[Passage, list[str]]]:
    document = Document(path)
    section = "Основной текст"
    url = f"https://drive.google.com/file/d/{file_id}/view"
    passages = []
    continuations = {}
    continuation = None
    for index, node in enumerate(document.element.body, 1):
        if node.tag.endswith("}p"):
            paragraph = Paragraph(node, document)
            value = " ".join(paragraph.text.split())
            heading = paragraph.style.name.startswith("Heading")
            number = re.match(r"^(\d+(?:\.\d+)*)(?:[.)])?\s", value)
            if heading or number:
                continuation = None
            elif continuation is not None and value:
                continuation.append(value)
            if heading and value:
                section = value[:150]
                continue
            if len(value) >= 35 and re.search(r"[.!?;:]$", value):
                location = f"п. {number[1]}" if number else section
                passage = Passage(source, f"{location} · блок {index}", value, url)
                passages.append(passage)
                if number and value.endswith(":"):
                    continuation = continuations[passage] = []
        elif node.tag.endswith("}tbl"):
            table = Table(node, document)
            header = ""
            headings = []
            for row_index, row in enumerate(table.rows, 1):
                raw_cells = [" ".join(cell.text.split()) for cell in row.cells]
                cells = list(dict.fromkeys(raw_cells))
                value = " | ".join(cell for cell in cells if cell)
                if row_index == 1 and value:
                    header = value[:160]
                    headings = raw_cells
                if continuation is not None and value:
                    if row_index == 1 or len(headings) != len(raw_cells) or not all(headings):
                        continuation.append(value)
                    else:
                        continuation.append(" | ".join(dict.fromkeys(
                            f"{title}: {cell}" for title, cell in zip(headings, raw_cells) if cell
                        )))
                if len(value) >= 35:
                    location = f"{section} · таблица {index}, строка {row_index}"
                    if header and row_index > 1:
                        location += f" ({header})"
                    passages.append(Passage(source, location, value, url))
    return passages, continuations


class Catalog:
    def __init__(self, folder: Path, model_dir: Path | None = None):
        self.documents = {}
        self.passages = []
        self.continuations = {}
        for filename, (label, file_id) in SOURCES.items():
            path = folder / filename
            if path.is_file():
                self.documents[filename] = label
                passages, continuations = read_passages(path, label, file_id)
                self.passages.extend(passages)
                self.continuations.update(continuations)
        self.index = [(passage, terms(passage.quote), terms(passage.section)) for passage in self.passages]
        self.frequency = Counter(term for _, body, section in self.index for term in body | section)
        self.encoder = None
        self.vectors = None
        if model_dir is not None and self.passages:
            from semantic import Encoder

            self.encoder = Encoder(model_dir)
            self.vectors = self.encoder.encode(
                [f"{p.source}. {p.section}. {p.quote}" for p in self.passages], "passage"
            )

    def semantic_scores(self, question: str):
        if self.encoder is None:
            return None
        vector = self.encoder.encode([question], "query")[0]
        return self.vectors @ vector

    def find(self, question: str, scope: str | None = None, condition: str | None = None) -> dict:
        parts = [part.strip().rstrip("?") + "?" for part in COMPOUND.split(question) if part.strip()]
        if len(parts) > 1:
            return {
                "state": "clarify",
                "prompt": "В сообщении несколько вопросов. Выберите, с какого начать.",
                "questions": parts,
            }
        recovery = self.rejected_project(question, scope)
        if recovery:
            return recovery
        clarified = self.cost_discrepancy(question, scope, condition)
        if clarified:
            return clarified
        query = terms(question)
        count = len(self.index)
        weights = {term: 1 + log((count + 1) / (self.frequency[term] + 1)) for term in query}
        total = sum(weights.values())
        if not total:
            return {"state": "missing"}
        scores = self.semantic_scores(question)
        numbers = set(re.findall(r"\b\d+\b", question))
        lower = question.casefold()
        asks_role = bool(re.search(r"\b(кто|кому)\b", lower))
        asks_time = bool(re.search(r"какой срок|за сколько|через сколько|в какие сроки|как часто", lower))
        role_action = next((action for action in ("выбор", "назнач")
                            if (asks_role or asks_time) and action in query), None)
        subject = re.search(r"гостиниц\w*|отпуск\w*|командировк\w*", lower)
        asks_contractor_access = "доступ" in query and bool(re.search(r"подрядчик", lower))
        matches = {}
        scope_hints = {}
        for index, (passage, body, section) in enumerate(self.index):
            if scope and passage.source != scope:
                continue
            if role_action and role_action not in body:
                continue
            if asks_time and not re.search(r"\d+\s*(?:рабоч\w*|календар\w*|дн|час|мин|недел|месяц)|"
                                           r"не позднее|до начала|не реже|в течение|после окончания",
                                           passage.quote, re.IGNORECASE):
                continue
            if subject and not terms(subject[0]) & body:
                continue
            if asks_contractor_access and not re.search(r"подрядчик", passage.quote, re.IGNORECASE):
                continue
            quote_body = re.sub(r"^\d+(?:\.\d+)*[.)]?\s+", "", passage.quote)
            if numbers and not numbers <= set(re.findall(r"\b\d+\b", quote_body)):
                continue
            overlap = query & (body | section)
            score = sum(weights[term] for term in overlap) / total
            body_overlap = query & body
            lexical = len(body_overlap) >= 2 and score >= 0.72
            similarity = float(scores[index]) if scores is not None else 0
            if not scope and len(body_overlap) >= 2 and similarity >= 0.84:
                scope_hints[passage.source] = max(similarity, scope_hints.get(passage.source, 0))
            semantic = bool(scope and body_overlap and similarity >= 0.82 and score >= 0.25)
            if not (lexical or semantic):
                continue
            ranked_score = score + 0.1 * len(body_overlap) / len(query)
            if scores is not None and scope:
                ranked_score = 0.75 * similarity + 0.25 * ranked_score
            matches.setdefault(passage.source, []).append((ranked_score, passage))
        for values in matches.values():
            values.sort(key=lambda item: item[0], reverse=True)
        candidates = sorted(matches, key=lambda source: matches[source][0][0], reverse=True)
        if not candidates:
            if len(scope_hints) > 1:
                return {"state": "clarify", "options": sorted(scope_hints, key=scope_hints.get, reverse=True)}
            return {"state": "missing"}
        best = matches[candidates[0]][0][0]
        close = [source for source in candidates if best - matches[source][0][0] <= 0.15]
        if len(close) > 1:
            return {"state": "clarify", "options": close}
        passage = matches[close[0]][0][1]
        evidence = [asdict(passage)]
        content = self.continuations.get(passage)
        if content and LIST_REQUEST.search(question):
            evidence[0]["quote"] = "\n".join([passage.quote, *content])
        return {"state": "answer", "evidence": evidence}

    def rejected_project(self, question: str, scope: str | None):
        source = SOURCES["ib-project.docx"][0]
        if scope not in (None, "", source):
            return None
        query = question.casefold().replace("ё", "е")
        if not (
            re.search(r"проект|карточк", query)
            and re.search(r"отклонил|отклонен", query)
            and re.search(r"как (поступ|действ)|что (делать|поменять)|верн|восстанов|возобнов|снова", query)
            and not re.search(r"\d|не\s+отклон|без\s+отклон", query)
        ):
            return None
        pair = next(
            (
                (previous, following)
                for previous, following in zip(self.passages, self.passages[1:])
                if previous.source == following.source == source
                and previous.section.startswith("п. 2.3 ·")
                and following.section.startswith("п. 2.4 ·")
                and "статус «Отклонён»" in previous.quote
                and "вернуть проект в работу" in following.quote
                and "статус на «Новый»" in following.quote
            ),
            None,
        )
        if pair is None:
            return None
        if not scope:
            return {
                "state": "clarify",
                "prompt": "Речь об отклонённой карточке проекта Дирекции ИБ?",
                "options": [source],
            }
        previous, following = pair
        return {"state": "answer", "evidence": [asdict(following), asdict(previous)],
                "process": "rejected_project"}

    def cost_discrepancy(self, question: str, scope: str | None, condition: str | None):
        source = SOURCES["ib-project.docx"][0]
        if scope != source:
            return None
        query = question.casefold().replace("ё", "е")
        if (not re.search(r"расхожден|несоответств", query) or "себестоим" not in query
                or not re.search(r"что делать|как поступ|что происходит|как обработ|как действ", query)
                or re.search(r"\d", query)):
            return None
        passage = next(
            (p for p in self.passages if p.source == source and p.section.startswith("п. 3.5 ·")
             and "Если несоответствия влекут" in p.quote),
            None,
        )
        if passage is None:
            return None
        if condition is None:
            stated = re.search(r"себестоимост[ьи]\s+(не\s+)?(?:меняется|изменяется)", query)
            if stated:
                condition = "Не меняют" if stated[1] else "Меняют"
        if condition is None:
            return {
                "state": "clarify",
                "prompt": "Расхождения меняют себестоимость собственных ресурсов проекта?",
                "conditions": ["Меняют", "Не меняют", "Не знаю"],
            }
        if condition == "Не знаю":
            return {"state": "missing"}
        if condition not in {"Меняют", "Не меняют"}:
            return {"state": "missing"}
        start = "Если несоответствия " + ("влекут" if condition == "Меняют" else "не влекут")
        sentence = next((line for line in passage.quote.split(". ") if line.startswith(start)), None)
        if not sentence:
            return {"state": "missing"}
        return {"state": "answer", "evidence": [asdict(passage) | {"quote": sentence.rstrip(".") + "."}]}
