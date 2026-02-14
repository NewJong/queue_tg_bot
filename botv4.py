import asyncio
import os
import aiohttp
import pandas as pd
import re
from dotenv import load_dotenv
from lingua import Language, LanguageDetectorBuilder
from aiogram import Bot, Dispatcher, types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
load_dotenv()

TG_TOKEN = os.getenv("TG_TOKEN")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")
admin_raw = os.getenv("ADMIN_IDS", "")
ADMIN_IDS = {int(i.strip()) for i in admin_raw.split(",") if i.strip()}

if not TG_TOKEN or not DISCORD_WEBHOOK:
    raise RuntimeError("TG_TOKEN или DISCORD_WEBHOOK не заданы")

AUTO_REPLY = {
    "ru": "Здравствуйте! Мы работаем над вашим запросом",
    "uk": "Вітаємо! Ми працюємо над вашим запитом",
    "en": "Hello! We’re working on your request",
}
CALL_KEYWORDS = {
    "ru": [
        "позвоните", "наберите меня", "наберите", "перезвоните",
        "можете позвонить", "позвоните мне", "набери меня пожалуйста",
        "позвоните пожалуйста", "позвони мне пожалуйста", "позвони", 
        "пазваните", "пазваните мне", "пазвони мне", "пазвони мне",
        "пазвани мне", "пазвани мне пожалуйста",
    ],
    "uk": [
        "наберіть мене", "подзвоніть", "можете подзвонити",
        "передзвоніть", "зателефонуйте", "наберіть"
    ],
    "en": [
        "call me", "please call", "can you call",
        "give me a call", "call please", "call", "call me please"
    ]
}
CALL_REPLY = {
    "ru": "Здравствуйте! Пару минут, пожалуйста, и мы Вас наберём",
    "uk": "Вітаємо! Декілька хвилин, будь ласка, і ми Вам зателефонуємо",
    "en": "Hello! Please give us a couple of minutes and we will call you"
}

CLOSE_KEYWORDS = [
    "done", "all done", "have a good day", "stay safe", "ready", "shift started", "fixed", "safe trip",
    "have a nice day", "have a good rest", "updated pickup time", "started the shift", "have a nice trip",
    "have a good one", "have a great day", "fixed your log", "unfortunately", "made a split", "made a cycle reset",
    "violations fixed", "fixed violations", "time added", "BOL added", "added some time", "have a safe trip", 
    "shift opened", "all the best", "all fixed", "log fixed", "shift available", "split activated", "request completed",
    "co-driver drop", "shift reopened", "new shift opened", "info added", "have a nice rest", "have a great rest",
    "log fixed", "logbook updated", "logbook fixed", "shift available", "started shift", "information added", "added information", 
    "added info", "added time", "added break", "break added", "PTI added", "added PTI",
    "готово", "хорошей дороги", "безопасной дороги", "хорошего дня", "всего наилучшего",
    "всего доброго", "новая смена доступна", "новая смена открыта", "смена открыта", "открыли смену", 
    "брейк добавлен", "к сожалению", "смена доступна", "поправили", "сделали сплит", "активировали сплит", 
    "сделали вам сплит", "запрос выполнен", "сделали сброс цикла", "добавили груз в логбук", "сделали брейк",
    "все исправили", "все поправили", "все готово", "хорошего дня и безопасной дороги", "хорошей и безопасной дороги",
    "добавили время", "добавили времени", "хорошего вам дня", "удачного вам дня", "начали смену", "смена началась",
    "добавили брейк", "добавили PTI", "PTI добавлен", "ПТИ добавлен", "ПТИ добавили", "сделали ПТИ",
    "готово", "гарної дороги", "гарного дня", "безпечної дороги", "на жаль", "відкрили зміну", "гарної дороги", 
    "всього найкращого", "зміна відкрита", "зробили", "додали", "зміна доступна", "сделали сплит", 
    "активували спліт", "зробили спліт", "зробили вам спліт", "запит виконано", "все готово", 
    "зробили скидання циклу", "нова зміна відкрита", "додали часу", "усе готово", "час додано",
    "гарного дня та безпечної дороги", "гарної та безпечної дороги", "вдалого вам дня", "гарного вам дня",
    "зробили рестарт циклу", "зробили вам рестарт циклу", "рестарт циклу зроблено", "цикл оновлено",
    "PTI додано", "додали PTI", "ПТІ додано", "додали ПТІ", "ПТИ додано", "додали ПТИ", 
]

THRESHOLD = 0.35 
FUZZY_THRESHOLD = 80
open_tasks = {}
pending_media_checks = {}
MEDIA_CHECK_DELAY = 30

BOT_START_TIME = datetime.utcnow()

df = pd.read_csv("data.csv")
texts = df["Text"].astype(str).tolist()

vectorizer = TfidfVectorizer(ngram_range=(1, 2),
    min_df=2, max_df=0.9
)

X = vectorizer.fit_transform(texts)

def needs_help(message: str, threshold: float = THRESHOLD):
    vec = vectorizer.transform([message])
    sims = cosine_similarity(vec, X)[0]
    max_sim = sims.max()
    return max_sim > threshold, max_sim

languages = [Language.ENGLISH, Language.RUSSIAN, Language.UKRAINIAN]

detector = LanguageDetectorBuilder.from_languages(*languages).build()

def detect_lang(text: str) -> str:
    lang = detector.detect_language_of(text or "")
    if not lang:
        return "en"
    return lang.iso_code_639_1.name.lower()

def is_call_request(text: str, lang: str) -> bool:
    text = text.lower()
    for phrase in CALL_KEYWORDS.get(lang, []):
        if phrase in text:
            return True
    return False

bot = Bot(token=TG_TOKEN)
dp = Dispatcher()

async def send_to_discord(text: str):
    async with aiohttp.ClientSession() as session:
        await session.post(DISCORD_WEBHOOK,json={"content": text})

async def handle_admin_command(msg: types.Message):
    text = (msg.text or "").strip()

    if text == "/tasks":
        if not open_tasks:
            await msg.answer("There are no open tasks")
            return

        now = datetime.utcnow()
        lines = ["Open tasks:"]

        for task in open_tasks.values():
            delta = now - task["opened_at"]
            minutes = int(delta.total_seconds() // 60)
            seconds = int(delta.total_seconds() % 60)

            lines.append(
                f"{task['title']} — open {minutes} min {seconds} sec"
            )

        await msg.answer("\n".join(lines))
        return

    if text.startswith("!done"):
        name = text.replace("!done", "").strip().lower()

        if not name:
            await msg.answer("Use: `!done group_name`")
            return

        for chat_id, task in list(open_tasks.items()):
            if name in task["title"].lower():
                del open_tasks[chat_id]
                if chat_id in pending_media_checks:
                    pending_media_checks[chat_id].cancel()
                    pending_media_checks.pop(chat_id, None)
                await msg.answer(f"Task {task['title']} closed prematurely")
                return

        await msg.answer("Task not found")
        return

    await msg.answer(
        "⚙️ Commands:\n"
        "/tasks — task list\n"
        "!done <group name> — close the task"
    )

@dp.message()
async def on_message(msg: types.Message):
    if msg.date.replace(tzinfo=None) < BOT_START_TIME:
        return
    if msg.chat.type == "private":
        if msg.from_user.id not in ADMIN_IDS:
            return
        await handle_admin_command(msg)
        return

    if msg.chat.type not in ("group", "supergroup"):
        return

    chat_id = msg.chat.id
    chat_title = msg.chat.title or "No Title"
    text = msg.text or msg.caption or ""
    is_media = bool(msg.photo or msg.voice or msg.document 
                    or msg.video or msg.video_note or msg.animation
                    or msg.audio or msg.sticker or msg.poll 
                    or msg.contact or msg.location or msg.venue)
    
    if msg.from_user.id in ADMIN_IDS:
        normalized_text = re.sub(r"[^a-zA-Zа-яА-ЯёЁіІїЇєЄ\s]", "", text).lower()
        
        for phrase in CLOSE_KEYWORDS:
            if phrase.lower() in normalized_text:
                if chat_id in open_tasks:
                    del open_tasks[chat_id]
                if chat_id in pending_media_checks:
                    pending_media_checks[chat_id].cancel()
                    pending_media_checks.pop(chat_id, None)
                return
        return
    
    need_help = False
    lang = detect_lang(text)

    is_call = is_call_request(text, lang) if text.strip() else False
    if text.strip() and not is_call:
        need_help, _ = needs_help(text)
    
    if is_call or need_help:
        if chat_id in open_tasks:
            return
        open_tasks[chat_id] = {
            "title": chat_title,
            "opened_at": datetime.utcnow(),
            "notifications_sent": [],
        }
        await msg.answer(CALL_REPLY.get(lang) if is_call else AUTO_REPLY.get(lang))
        await send_to_discord(msg.chat.title)
        if chat_id in pending_media_checks:
            pending_media_checks[chat_id].cancel()
            pending_media_checks.pop(chat_id, None)
        return

    if is_media:
        if chat_id not in pending_media_checks:
            async def media_check():
                try:
                    await asyncio.sleep(MEDIA_CHECK_DELAY)
                    if chat_id not in open_tasks:
                        open_tasks[chat_id] = {
                            "title": chat_title,
                            "opened_at": datetime.utcnow(),
                            "notifications_sent": [],
                        }
                        await send_to_discord(f"**{chat_title} needs an answer**")
                except asyncio.CancelledError:
                    pass
                finally:
                    pending_media_checks.pop(chat_id, None)

            task = asyncio.create_task(media_check())
            pending_media_checks[chat_id] = task

async def monitor_tasks():
    while True:
        now = datetime.utcnow()
        for chat_id, task in list(open_tasks.items()):
            group_name = task.get("title")
            elapsed_minutes = (now - task["opened_at"]).total_seconds() / 60
            milestones = [20, 40]

            for minutes in milestones:
                if elapsed_minutes >= minutes and minutes not in task["notifications_sent"]:
                    await send_to_discord(f"**{group_name} task open for more than {minutes} minutes**")
                    task["notifications_sent"].append(minutes)

            if now - task["opened_at"] > timedelta(hours=1):
                del open_tasks[chat_id]

        await asyncio.sleep(120)

async def main():
    asyncio.create_task(monitor_tasks())
    print("Bot started")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("Bot stopped")