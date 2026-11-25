"""Проверки реплик из LOGII.txt без микрофона, синтезатора и команд роботу."""

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import phrase_gate
import voice_assistant as voice
from valera_memory import ConversationMemory


class DialogueTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = str(Path(self.folder.name) / "memory.jsonl")
        with patch.object(ConversationMemory, "_import_questions_log"):
            memory = ConversationMemory(self.path)
        bot = object.__new__(voice.VoiceAssistant)
        bot._processing_lock = threading.Lock()
        bot._chat_id = bot._memory_chat_id = "guest"
        bot._conversation_history = []
        bot._last_effective = bot._last_topic = ""
        bot._last_mirea = False
        bot._force_phrase = bot._no_clean = False
        bot._pending_raw = []
        bot._pending_followup = bot.noise_profile = bot._controller = None
        bot._collection_id = "mirea"
        bot.memory = memory
        bot.gate = Mock(enabled=True, digest="")
        bot.gate.is_complete.return_value = {"complete": True}
        bot.gate.detect_offered_topic.return_value = ""
        bot.gate.discussion_items.return_value = ()
        bot.runtime = Mock()
        bot.runtime.feature.side_effect = lambda name, **kwargs: name == "mirea_collection"
        bot.runtime.fire_triggers.return_value = []
        bot.rag = Mock()
        bot.rag.get_llm_provider.return_value = "deepseek"
        bot._get_collection_id = Mock(return_value="mirea")
        bot._rag_query = Mock()
        bot._get_asr = Mock()
        bot._speak = Mock()
        for target, kwargs in ((voice, {"clean_audio": lambda a, *args: a, "log_question": Mock()}),
                               (voice.sf, {"write": Mock()})):
            context = patch.multiple(target, **kwargs)
            context.start()
            self.addCleanup(context.stop)
        voice.set_answer_context("")
        self.bot = bot

    def say(self, text, answer=None):
        if answer:
            self.bot._rag_query.return_value = {"response": answer, "contexts": [{"content": answer}]}
            self.bot.rag.direct_answer.return_value = answer
        self.bot._get_asr.return_value.transcribe.return_value.full_text = text
        self.bot._speak.reset_mock()
        self.bot._handle_phrase_inner(np.zeros(4))
        self.bot._speak.assert_called_once()
        return self.bot._speak.call_args.args[0]

    def test_institute_tour(self):
        self.say("Расскажи про институты по порядку.", "В ИИТ занимаются разработкой программ.")
        self.say("Ну так давай.", "В ИИИ изучают робототехнику и системы управления.")
        self.assertIn("Институт искусственного интеллекта", self.bot._rag_query.call_args.kwargs["query_text"])
        self.bot.memory = ConversationMemory(self.path)
        self.say("Давай.", "В ИКБ изучают информационную безопасность.")
        self.assertIn("Институт кибербезопасности", self.bot._rag_query.call_args.kwargs["query_text"])
        detail = self.say("А что там изучают?", "В ИКБ изучают безопасность программных решений.")
        self.assertIn("ИКБ", self.bot._rag_query.call_args.kwargs["query_text"])
        self.assertEqual(self.say("Повтори."), detail)
        self.say("Дальше.", "В ИРИП изучают радиоэлектронику.")
        self.assertIn("ИРИП", self.bot._rag_query.call_args.kwargs["query_text"])
        self.bot.rag.direct_answer.assert_not_called()

    def test_admission_and_program(self):
        self.say("Институтов.", "В ИИТ занимаются разработкой программ.")
        self.assertIn("программы", self.say("Расскажи про баллы поступления и стоимость обучения."))
        self.say("Прикладная информатика.", "Для этой программы условия зависят от года приёма.")
        query = self.bot._rag_query.call_args.kwargs["query_text"]
        for part in ("высшее образование", "Прикладная информатика", "ИИТ", "стоимость"):
            self.assertIn(part, query)
        self.say("А сколько стоит?", "Точную стоимость обучения нужно уточнить.")
        self.assertIn("Прикладная информатика", self.bot._rag_query.call_args.kwargs["query_text"])

    def test_open_day_admission_conversation(self):
        self.say("Кто ты и что ты умеешь? Я пришёл на день открытых дверей. Какие есть направления? "
                 "С моими 240 баллами куда я могу поступить?", "Я Валера, помогаю выбрать направление.")
        self.assertIn("Подбор направления", self.bot._rag_query.call_args.kwargs["query_text"])
        self.say("Давай что-то связанное с роботами и дронами.",
                 "Рассмотрите направление «Мехатроника и робототехника».")
        self.assertIn("беспилотные системы", self.bot._rag_query.call_args.kwargs["query_text"])
        self.say("Расскажи мне про направление. Сколько баллов, как поступить, что требуется, сколько стоимость? "
                 "Всё подряд про направление. Про лабораторию я тебя не спрашивал.",
                 "В каталоге указаны результаты конкурса по каждому профилю.")
        query = self.bot._rag_query.call_args.kwargs["query_text"]
        for part in ("Мехатроника и робототехника", "робототехника", "Баллы гостя: 240", "что требуется", "стоимость"):
            self.assertIn(part, query)
        self.bot.gate.discussion_items.assert_not_called()
        self.assertEqual(self.bot._rag_query.call_count, 3)
        self.assertEqual(self.say("Какие направления связаны с боевыми дронами?"), voice.FORBIDDEN_REFUSAL)
        self.assertEqual(self.bot._rag_query.call_count, 3)

    def test_program_choice_and_context_boundaries(self):
        from valera_dialogue import named_program, selection_topic
        self.assertEqual(named_program("Хочу выбрать направление себе в вузе."), "")
        topic = selection_topic("Я хочу поступить, увлекаюсь робототехникой.", "", "")
        self.assertTrue(topic)
        self.assertEqual(selection_topic("Сколько стоит дрон?", topic, ""), "")
        self.assertEqual(selection_topic("Кто ты?", topic, ""), "")
        self.assertEqual(selection_topic("Давай что-то связанное с роботами.", "", ""), "")
        self.assertIn("бэкендом", voice.fix_asr("А есть что-то между викендом, робототехникой и IT?"))
        self.assertIn("викендом", voice.fix_asr("Перед викендом."))

    def test_selected_offer_and_institute_order(self):
        self.say("Ну давай по очереди про каждый институт.",
                 "В ИИТ изучают разработку программ. Хотите узнать подробнее о его программах?")
        self.say("Давай.", "В ИИТ есть направление «Программная инженерия».")
        query = self.bot._rag_query.call_args.kwargs["query_text"]
        self.assertIn("ИИТ", query)
        self.assertIn("Назови программы", query)
        self.say("Дальше.", "В ИИИ изучают мехатронику и робототехнику.")
        self.assertIn("ИИИ", self.bot.memory.last()["topic"])
        self.bot.memory.add("Почему Институт ИИ?", "Вам подойдёт ИИИ. Вас больше привлекает разработка "
                            "алгоритмов или создание самих роботов?", "guest", True)
        self.say("Давай про роботов.", "Подойдёт мехатроника и робототехника.")
        query = self.bot._rag_query.call_args.kwargs["query_text"]
        self.assertIn("Гость выбрал: Давай про роботов", query)
        self.assertIn("ИИИ", query)
        self.bot.gate.detect_offered_topic.assert_not_called()

    def test_overview_supersedes_pending_program(self):
        self.say("Институтов.", "В ИИТ изучают разработку программ.")
        self.say("Расскажи про баллы поступления и стоимость обучения.")
        self.assertTrue(self.bot._pending_followup.awaiting_program)
        self.say("Я же сказал, расскажи про все институты. Что ты тупишь?",
                 "В университете шесть институтов.")
        query = self.bot._rag_query.call_args.kwargs["query_text"]
        self.assertIn("Обзор институтов", query)
        self.assertNotIn("Образовательная программа:", query)
        self.say("Расскажи про институты, какие есть в вузе, все виды направлений и средний балл для поступления.",
                 "Институты различаются по направлениям подготовки.")
        self.assertIn("средний", self.bot._rag_query.call_args.kwargs["query_text"])
        self.assertIsNone(self.bot._pending_followup)

    def test_equipment_to_admissions(self):
        self.bot.memory.add("Какие роботы есть в лаборатории?", "Есть роботы Юнитри.", "guest", True)
        for q in ("Расскажи про каждую модель.", "Расскажи про собак.",
                  "В чём разница между Goed один и Go 2?"):
            self.say(q, "Можно подробнее обсудить четвероногих роботов.")
            self.assertIn("лаборатории", self.bot._rag_query.call_args.kwargs["query_text"])
        self.assertIn("Go1 и Go2", self.bot._rag_query.call_args.kwargs["query_text"])
        self.say("Какие направления больше всего подходят для того, чтобы работать в этой лаборатории?",
                 "Рассмотрите мехатронику и робототехнику.")
        self.assertIn("Интересы: робототехника", self.bot._rag_query.call_args.kwargs["query_text"])
        self.assertEqual(voice.spoken_numbers("Go1 и Go2"), "Го один и Го два")
        self.assertEqual(voice.words_only("ИКБ, ИТУ, А-Б-Б Го-Фа."), "институт кибербезопасности и цифровых технологий институт технологий управления а бэ бэ серии Гофа")

    def test_incomplete_stream_is_not_read_as_an_answer(self):
        bot = self.bot
        bot._tts_speaker = "test"
        bot._interrupt_tts = False
        bot._synthesize_pcm_bytes = Mock(return_value=b"pcm")
        with patch.object(voice, "play_audio", return_value=True):
            bot.rag.chat_query_stream.return_value = iter([{"t": "d", "x": "В"}, {"t": "done", "response": "В"}])
            self.assertEqual(bot._stream_answer("Какие роботы там есть?", "mirea"), "")
            bot._synthesize_pcm_bytes.assert_not_called()
            for answer in ("В лаборатории есть роботы.", "Да.", "Нет.", "Пять."):
                bot.rag.chat_query_stream.return_value = iter([{"t": "d", "x": answer[:1]},
                                                               {"t": "done", "response": answer}])
                self.assertEqual(bot._stream_answer("Вопрос", "mirea"), answer)

    def test_guest_correction_and_isolation(self):
        self.say("Валера, запомни, что мегалаборатория находится в ГЭ.")
        self.assertEqual(self.say("Нет, находится в Г100."), "Понял, в Г100.")
        self.bot.memory = ConversationMemory(self.path)
        self.assertEqual(self.say("Окей, где находится мегалаборатория?"),
                         "Вы сказали, что мегалаборатория находится в Г100.")
        self.bot._rag_query.assert_not_called()
        self.assertEqual(self.bot.memory.guest_fact_reply("Где находится мегалаборатория?", "other"), ("", None))

    def test_menu_and_example(self):
        self.bot.memory.add("Что можно посмотреть?", "Можно осмотреть кампусы, лаборатории или музеи.", "guest", True)
        self.bot.gate.discussion_items.return_value = ("кампусы", "лаборатории", "музеи")
        self.say("Расскажи про всё.", "Кампусы находятся в разных районах города.")
        self.assertIn("кампусы", self.bot._rag_query.call_args.kwargs["query_text"])
        self.say("Ну, например.", "Главный кампус находится на проспекте Вернадского.")
        self.assertIn("кампусы", self.bot._rag_query.call_args.kwargs["query_text"])
        self.say("Продолжай.", "В лаборатории работают с мобильными роботами.")
        self.assertIn("лаборатории", self.bot._rag_query.call_args.kwargs["query_text"])

    def test_extract_menu(self):
        gate = phrase_gate.PhraseGate()
        for result, expected in [('["кампусы","музеи"]', ("кампусы", "музеи")),
                                 ('["музеи","кампусы"]', ()), ('["кампусы","метро"]', ())]:
            with patch.object(phrase_gate, "_post_chat", return_value=result):
                self.assertEqual(gate.discussion_items("Можно осмотреть кампусы или музеи."), expected)

    def test_new_topic_and_missing_sources(self):
        self.bot._rag_query.return_value = {"response": "Случайная догадка.", "contexts": []}
        self.assertIn("нет подтверждённых сведений", self.say("Расскажи про институты по порядку."))
        self.say("Дальше.", "В ИИИ изучают робототехнику и системы управления.")
        self.assertIn("Институт искусственного интеллекта", self.bot._rag_query.call_args.kwargs["query_text"])
        self.bot.rag.direct_answer.assert_not_called()
        self.bot.gate.discussion_items.return_value = ("сравнение", "перестановка")
        self.say("Объясни пузырьковую сортировку по шагам.", "Сначала сравниваем два соседних элемента.")
        row = self.bot.memory.last()
        self.assertIn("сортировку", row["eq"])
        self.assertNotIn("ИИИ", row["eq"])
        self.say("Продолжай.", "Затем меняем соседние элементы местами.")
        self.assertIn("перестановка", self.bot.memory.last()["eq"])

    def test_legacy_history(self):
        memory = self.bot.memory
        memory.add("Запомни, что мегалаборатория находится в ГЭ.", "Запомнил.", "guest")
        memory.add("Нет, находится в Г100.", "Понял, в корпусе Г100.", "guest")
        answer, _ = memory.guest_fact_reply("Адрес мегалаборатории?", "guest")
        self.assertEqual(answer, "Вы сказали, что мегалаборатория находится в Г100.")
        memory.add("Расскажи про институты.", "В ИИТ изучают разработку программ. "
                   "В ИИИ изучают робототехнику. Перейти к следующему институту?", "guest", True)
        self.say("Давай.", "В ИКБ изучают информационную безопасность.")
        self.assertIn("ИКБ", self.bot._rag_query.call_args.kwargs["query_text"])

    def test_names_abbreviations_and_spoken_numbers(self):
        self.assertEqual(voice.clean_question("Лера!"), "")
        self.assertEqual(voice.fix_asr(voice.clean_question("Лера, что такое и ка бэ?")),
                         "что такое ИКБ?")
        self.assertEqual(voice.clean_question("Где галерея?"), "Где галерея?")
        answer = voice.finalize_answer("С 01.09.2026 стоимость от 21 000 ₽.", 80)
        self.assertEqual(answer, "С первого сентября две тысячи двадцать шестого года стоимость от двадцати одной тысячи рублей.")
        self.assertEqual(voice.finalize_answer("Версия 1.5.", 20), "Версия один точка пять.")

    def test_punctuation_is_removed_only_at_synthesis(self):
        bot = self.bot
        bot._tts_speaker = "test"
        bot._interrupt_tts = False
        raw = "Вы выбрали «Автономные роботы»? [эмоция:радость] Да, всё верно! [спокойствие]"
        bot.rag.chat_query_stream.return_value = iter({"t": "d", "x": ch} for ch in raw)
        bot.runtime.get.side_effect = lambda section, key, default=None: default
        expected = ["Вы выбрали Автономные роботы", "Да всё верно"]
        with patch.object(voice, "get_xtts_client") as client, patch.object(voice, "play_audio", return_value=True):
            synth = client.return_value.synthesize
            synth.return_value = b"\x10\x00" * 1024
            displayed = bot._stream_answer("Проверка", "collection")
            self.assertEqual(displayed, "Вы выбрали «Автономные роботы»? Да, всё верно!")
            self.assertEqual([c.args[0] for c in synth.call_args_list], expected)
            self.assertNotEqual(synth.call_args_list[0].kwargs["speed"], synth.call_args_list[1].kwargs["speed"])
            synth.reset_mock()
            bot.runtime.feature.side_effect = lambda name, **kwargs: name == "tts"
            bot._speak = voice.VoiceAssistant._speak.__get__(bot)
            with patch.object(voice, "TTS_PARALLEL_SENTENCES", True):
                bot._speak(raw)
            self.assertEqual([c.args[0] for c in synth.call_args_list], expected)
            synth.reset_mock()
            bot._speak("«?!…»", literal=True)
            synth.assert_not_called()
        bot.tts = Mock()
        with patch.object(bot, "_synthesize_pcm_bytes", return_value=None):
            bot._speak_whole_fallback("«Да», 2,5!", "test")
        self.assertEqual(bot.tts.speak.call_args.args[0], "Да две целых пять десятых")

    def test_stream_sends_separate_style_to_synthesizer(self):
        bot = self.bot
        bot._interrupt_tts = False
        bot._tts_speaker = "test"
        bot._synthesize_pcm_bytes = Mock(return_value=b"pcm")
        raw = "Всё готово. [эмоция:радость] Следующий пункт. [уверенность]"
        bot.rag.chat_query_stream.return_value = ({"t": "d", "x": ch} for ch in raw)
        with patch.object(voice, "play_audio") as playback:
            answer = bot._stream_answer("тема", "collection")
        self.assertEqual(answer, "Всё готово. Следующий пункт.")
        pairs = [(call.args[0], call.args[2]) for call in bot._synthesize_pcm_bytes.call_args_list]
        self.assertEqual(pairs, [("Всё готово.", voice.emotion_style("радость")),
                                 ("Следующий пункт.", voice.emotion_style("уверенность"))])
        self.assertEqual(playback.call_count, 2)

    def test_stalled_stream_answers_once_and_takes_next_question(self):
        bot = self.bot
        bot._tts_speaker = "test"
        bot._interrupt_tts = False
        bot._synthesize_pcm_bytes = Mock(return_value=b"pcm")
        bot._maybe_auto_new_chat = Mock()
        bot.runtime.feature.side_effect = lambda name, **_: name in {"mirea_collection", "tts", "stream_tts"}
        release = threading.Event()
        self.addCleanup(release.set)

        def stalled(*args, **kwargs):
            yield {"t": "meta"}
            yield {"t": "d", "x": "Обрывок"}
            release.wait(2)
            yield {"t": "done", "response": "Запоздалый ответ."}

        bot.rag.chat_query_stream.side_effect = stalled
        question = "Расскажи про университет."
        bot._get_asr.return_value.transcribe.return_value.full_text = question
        with patch.object(voice, "VOICE_SEARCH_WAIT", 0.05):
            bot._handle_phrase(np.zeros(4))
        reply = bot._speak.call_args.args[0]
        self.assertIn("университет", reply.lower())
        self.assertEqual(voice.log_question.call_args.kwargs["source"], "local-kb")
        bot.runtime.set_transcript.assert_called_with(question, reply)
        bot.runtime.set_status.assert_called_with("Слушаю…")
        self.assertFalse(bot._busy)
        self.assertEqual(bot._chat_id, "guest")
        bot._rag_query.assert_not_called()
        bot.rag.create_chat.assert_not_called()
        bot.rag.direct_answer.assert_not_called()
        bot._speak.assert_called_once()
        release.set()
        self.assertTrue(voice._VOICE_SEARCH_SLOT.acquire(timeout=1))
        voice._VOICE_SEARCH_SLOT.release()
        bot._synthesize_pcm_bytes.assert_not_called()

        bot.rag.chat_query_stream.side_effect = None
        bot.rag.chat_query_stream.return_value = iter([{"t": "done", "response": "Университет находится в Москве."}])
        with patch.object(voice, "play_audio", return_value=True) as play:
            bot._handle_phrase(np.zeros(4))
        play.assert_called_once()
        self.assertEqual(voice.log_question.call_args.kwargs["source"], "chat-stream")
        self.assertFalse(bot._busy)

    def test_interrupted_stream_keeps_complete_sentence(self):
        bot = self.bot
        bot._tts_speaker = "test"
        bot._interrupt_tts = False
        bot._synthesize_pcm_bytes = Mock(return_value=b"pcm")
        bot.rag.chat_query_stream.return_value = iter([
            {"t": "d", "x": "В университете изучают робототехнику. Затем"},
            {"t": "err", "message": "соединение потеряно"},
        ])
        with patch.object(voice, "play_audio", return_value=True) as play:
            self.assertEqual(bot._stream_answer("вопрос", "mirea"), "В университете изучают робототехнику.")
        play.assert_called_once()

    def test_plain_query_preserves_chat_on_error_and_recreates_missing_chat(self):
        bot = self.bot
        del bot._rag_query
        bot.rag.chat_query.side_effect = TimeoutError("сервер занят")
        question = "Какова стоимость обучения в университете?"
        answer = self.say(question)
        self.assertIn("поиск недоступен", answer)
        bot.rag.chat_query.assert_called_once()
        self.assertEqual(voice.log_question.call_args.kwargs["source"], "search-unavailable")
        bot.rag.create_chat.assert_not_called()
        bot.rag.query.assert_not_called()
        self.assertEqual(bot._chat_id, "guest")
        self.assertEqual(bot._conversation_history, [])
        bot._save_chat_id = Mock()
        bot.rag.create_chat.return_value = "new"
        bot.rag.chat_query.side_effect = [voice.ChatMissing(), {"response": "Готово."}]
        self.assertEqual(bot._rag_query(question, "mirea")["response"], "Готово.")
        bot.rag.create_chat.assert_called_once()
        self.assertEqual(bot._chat_id, "new")

    def test_speech_and_style_across_chunk_boundaries(self):
        cases = (
            ("Привет. [эмоция:радость] До встречи. [эмоция:дружелюбие]",
             [("Привет.", "радость"), ("До встречи.", "дружелюбие")]),
            ("Готово. уверенность Буду помнить. дружелюбие.",
             [("Готово.", "уверенность"), ("Буду помнить.", "дружелюбие")]),
            ("Готово. [эмо", [("Готово.", None)]),
            ("Готово. эмо.", [("Готово.", None)]),
        )
        for text, expected in cases:
            for cut in range(len(text) + 1):
                buf = ""
                sentences = []
                for chunk in (text[:cut], text[cut:]):
                    ready, buf = voice.split_complete_sentences(buf + chunk)
                    sentences.extend(ready)
                if buf.strip():
                    sentences.append(buf)
                actual = [voice.parse_emotion(s) for s in sentences]
                self.assertEqual([pair for pair in actual if pair[0]], expected, (text, cut))
            self.assertEqual(voice.split_answer_for_tts(text),
                             [(s, voice.emotion_style(e)) for s, e in expected])
        self.assertEqual(voice.parse_emotion("Это уверенность в себе.")[0], "Это уверенность в себе.")


if __name__ == "__main__":
    unittest.main()
