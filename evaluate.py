"""Evaluate the four organizer-supplied documents against annotated employee questions."""

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from benchmark import DBRCT, IB, PRESALE, TECH
from retrieval import Catalog


# Expected document and clauses were assigned from the original DOCX before running the search.
CASES = [
    ("dbrct-01", DBRCT, "Без квалификационной анкеты заявку на пресейл возьмут в работу?", "answer", "1.1.1"),
    ("dbrct-02", DBRCT, "Кто отвечает за получение требований к результатам от заказчика?", "answer", "1.3"),
    ("dbrct-03", DBRCT, "За сколько рабочих дней до подачи предложения нужно инициировать заявку?", "answer", "1.5"),
    ("dbrct-04", DBRCT, "Кто проводит первичную валидацию заявки на пресейл?", "answer", "2.1"),
    ("dbrct-05", DBRCT, "По какому шаблону архитектор создаёт расчёт трудозатрат?", "answer", "3.2"),
    ("dbrct-06", DBRCT, "Куда архитектор направляет готовый расчёт трудозатрат на согласование?", "answer", "5.1"),
    ("ib-01", IB, "Кто назначает РП после уведомления из CRM?", "answer", "2.1"),
    ("ib-02", IB, "Когда нужно запросить РП относительно согласования договора?", "answer", "1.4"),
    ("ib-03", IB, "Через сколько рабочих дней отклоняют карточку, если SS не ответил?", "answer", "2.3"),
    ("ib-04", IB, "Проект отклонили. Как снова запустить его в работу?", "answer", "2.4"),
    ("ib-05", IB, "Что поменять в CRM, чтобы восстановить отклонённую карточку проекта?", "answer", "2.4"),
    ("ib-06", IB, "Разрешён ли доступ субподрядчика к сайту проекта?", "answer", "4.3"),
    ("ib-07", IB, "Что делать, если нашли несоответствия в себестоимости проекта?", "clarify", None),
    ("tech-01", TECH, "Что делает TPS, когда для расчёта трудозатрат не хватает документов?", "answer", "2.6"),
    ("tech-02", TECH, "Кто подбирает субподрядчика при пресейле?", "answer", "2.11"),
    ("tech-03", TECH, "Какие сроки актуализации данных о потенциальных подрядчиках?", "answer", "2.12"),
    ("tech-04", TECH, "Можно ли изменить утверждённый расчёт трудозатрат, если объём сделки тот же?", "answer", "2.16"),
    ("tech-05", TECH, "Кому передают эскалацию о нарушении срока SLA?", "answer", "2.17"),
    ("tech-06", TECH, "Кто создаёт в CRM заявку на расчёт для спецификации?", "answer", "1.3"),
    ("pre-01", PRESALE, "Кто заносит первичные данные заказчика в возможную сделку CRM?", "answer", "1.2"),
    ("pre-02", PRESALE, "Можно ли дать официальное ТКП без согласования с пресейл-архитектором?", "answer", "2.3"),
    ("pre-03", PRESALE, "Как разделить заявки, если клиент просит и внедрение, и техподдержку?", "answer", "2.5"),
    ("pre-04", PRESALE, "На кого автоматически назначается открытая заявка на услуги?", "answer", "2.6"),
    ("pre-05", PRESALE, "Какой срок у ПА, чтобы взять назначенную заявку в работу?", "answer", "5.1"),
    ("pre-06", PRESALE, "Кто выставляет приоритет заявке на расчёт?", "answer", "3.3"),
    ("choice-01", None, "Какие действия выполняют при назначении руководителя проекта?", "clarify", None),
    ("gap-01", None, "Кто подписывает агентский договор с новым дистрибьютором?", "missing", None),
    ("gap-02", None, "Как оформить учебный отпуск?", "missing", None),
    ("gap-03", None, "Можно ли перевести активированную лицензию на другое юридическое лицо?", "missing", None),
    ("gap-04", IB, "Отклоняют ли карточку проекта через 43 рабочих дня без ответа от SS?", "missing", None),
    ("gap-05", TECH, "Кому оформить авансовый отчёт за поездку к подрядчику?", "missing", None),
    ("gap-06", PRESALE, "По какому номеру звонить в службу качества в выходные?", "missing", None),
]


def evaluate(catalog):
    counts = Counter()
    durations = []
    details = []
    for identifier, scope, question, expected, clause in CASES:
        before = perf_counter()
        result = catalog.find(question, scope)
        durations.append((perf_counter() - before) * 1000)
        evidence = result.get("evidence", [])
        state = result["state"]
        matches_clause = not clause or bool(evidence and evidence[0]["source"] == scope
                                           and evidence[0]["section"].startswith(f"п. {clause} ·"))
        if state == expected and matches_clause:
            outcome = "pass"
        elif state == "answer":
            outcome = "wrong_answer"
        else:
            outcome = "miss"
        counts[outcome] += 1
        details.append({
            "id": identifier, "scope": scope, "question": question,
            "expected": expected, "clause": clause, "actual": state, "outcome": outcome,
            "sources": [entry["section"] for entry in evidence],
        })
    return {"total": len(CASES), "counts": dict(counts), "median_ms": round(median(durations), 2),
            "cases": details}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path(__file__).resolve().parent.parent / "private-corpus")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    report = evaluate(Catalog(options.corpus, options.model))
    if options.output:
        options.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for record in report["cases"]:
        print(record["id"], record["outcome"], record["actual"], "/".join(record["sources"]))
    print({key: value for key, value in report.items() if key != "cases"})
