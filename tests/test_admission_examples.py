"""Проверки подбора профилей на строках каталога, используемого для заливки."""
import csv
import runpy
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "WebRAgent/app/services/admission_examples.py"))


class AdmissionExamplesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = []
        for path in (ROOT / "knowledge_base/rtu_mirea_llm_kb/data").glob("programs_*.csv"):
            with path.open(encoding="utf-8") as f:
                rows = ["; ".join(f"{k}={v}" for k, v in row.items()) for row in csv.DictReader(f)]
            cls.items.append(dict(title=path.name, content=" ".join(rows), payload={}))

    def test_guest_interest_and_price(self):
        history = [{"role": "user", "content": "Я увлекаюсь робототехникой."},
                   {"role": "assistant", "content": "Посмотрите полигон Город."}]
        for q in ("Какие решения мне подойдут и сколько это будет стоить на платке?",
                  "Какие направления мне подойдут для обучения и сколько это будет стоить на платке?"):
            self.assertTrue(module["asks_programs"](q))
            rows = module["examples"](self.items, q, history)
            self.assertEqual(len(rows), 2)
            self.assertIn("Автономные роботы", rows[0]["content"])
            self.assertIn("мультироторные", rows[1]["content"])
            self.assertTrue(all("420 000 рублей в год" in row["content"] for row in rows))

    def test_drone_profile_keeps_its_own_scores_and_price(self):
        q = ("Подбор направления в РТУ МИРЭА, высшее образование. Интересы: робототехника, беспилотные системы. "
             "Сколько баллов и какова стоимость?")
        self.assertTrue(module["asks_programs"](q))
        rows = module["examples"](self.items, q, [])
        first = rows[0]["content"]
        for part in ("15.03.06", "мультироторные", "2025 — 225", "2026 — 247", "420 000 рублей в год"):
            self.assertIn(part, first)
        self.assertNotIn("2026 — 265", first)

    def test_development_and_robotics_together(self):
        q = "Есть что-то между бэкендом и робототехникой? Чем занимаются на этом направлении?"
        rows = module["examples"](self.items, q, [])
        self.assertTrue(any("Киберфизические" in row["content"] for row in rows))

    def test_new_interest_and_level(self):
        history = [{"role":"user", "content":"Мне нравится робототехника."}]
        rows = module["examples"](self.items, "Какие программы по химии подойдут?", history)
        self.assertTrue(all("хим" in row["content"].lower() for row in rows))
        rows = module["examples"](self.items, "Куда поступить в магистратуру по робототехнике?", [])
        self.assertIn("Интеллектуальные мобильные роботы", rows[0]["content"])
        self.assertEqual(module["examples"](self.items, "Куда поступить в колледж?", []), [])
