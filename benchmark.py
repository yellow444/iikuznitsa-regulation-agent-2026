"""Small inspection set for the four organizer-supplied regulations."""

from argparse import ArgumentParser
from pathlib import Path
from statistics import median
from time import perf_counter

from retrieval import Catalog


DBRCT = "ДБРЦТ · проекты"
IB = "Дирекция ИБ · проекты"
TECH = "Техническая дирекция · проекты"
PRESALE = "Дирекция ИБ · пресейл"
CASES = [
    ("Может ли бюджетная оценка использоваться для подачи в конкурс?", DBRCT, "answer", "2.4"),
    ("Что происходит с заявкой без квалификационной анкеты?", DBRCT, "answer", "1.1.1"),
    ("Кому поручено выбирать подрядчика на этапе пресейла?", TECH, "answer", "2.11"),
    ("Кому поручают новую заявку на услуги?", PRESALE, "answer", "2.6"),
    ("Когда отклоняется карточка проекта при отсутствии ответа от SS?", IB, "answer", "2.3"),
    ("Как поступить, если проект отклонили?", IB, "answer", "2.4"),
    ("Кто назначает руководителя проекта?", None, "clarify", None),
    ("Что делать с расхождениями в себестоимости проекта?", IB, "clarify", None),
    ("Через 45 рабочих дней отклоняется карточка при отсутствии ответа от SS?", IB, "missing", None),
    ("Что делать, если ключ не пришёл, а оплата прошла?", None, "missing", None),
    ("Можно ли вернуть лицензию после её активации?", None, "missing", None),
    ("Кто подписывает агентский договор?", None, "missing", None),
]


def run(catalog):
    times = []
    correct = 0
    for question, scope, state, clause in CASES:
        start = perf_counter()
        result = catalog.find(question, scope)
        times.append((perf_counter() - start) * 1000)
        locations = [item["section"] for item in result.get("evidence", [])]
        passed = result["state"] == state and (
            not clause or any(location.startswith(f"п. {clause} ·") for location in locations)
        )
        correct += passed
        print(f"{'OK' if passed else 'MISS':4} {state:7} {result['state']:7} {', '.join(locations)} | {question}")
    print(f"{correct}/{len(CASES)} · median {median(times):.1f} ms")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path(__file__).resolve().parent.parent / "private-corpus")
    parser.add_argument("--model", type=Path)
    options = parser.parse_args()
    args = {"model_dir": options.model} if options.model else {}
    run(Catalog(options.corpus, **args))
