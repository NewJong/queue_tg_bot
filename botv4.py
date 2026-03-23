import asyncio
import os
import random
import aiohttp
import pandas as pd
import re
from dotenv import load_dotenv
from lingua import Language, LanguageDetectorBuilder
from aiogram import Bot, Dispatcher, types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from asyncio import Queue

load_dotenv()

TG_TOKEN = os.getenv("TG_TOKEN")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")
admin_raw = os.getenv("ADMIN_IDS", "")
ADMIN_IDS = {int(i.strip()) for i in admin_raw.split(",") if i.strip()}

session = None
discord_queue = Queue()

if not TG_TOKEN or not DISCORD_WEBHOOK:
    raise RuntimeError("TG_TOKEN или DISCORD_WEBHOOK не заданы")

EMOJIS = ["⏳", "🔎", "🛠️", "😊", "🤝"]

AUTO_REPLY = {
    "ru": [
        "Здравствуйте! Работаем над вашим запросом",
        "Здравствуйте! Приступаем к работе",
        "Здравствуйте! Обрабатываем ваш запрос",
        "Здравствуйте! Занимаемся вашим запросом",
        "Здравствуйте! Ваш запрос принят в обработку",
    ],
    "uk": [
        "Вітаємо! Працюємо над вашим запитом",
        "Вітаємо! Обробляємо ваш запит",
        "Вітаємо! Приступаємо до роботи",
        "Вітаємо! Працюємо над цим",
        "Вітаємо! Опрацьовуємо ваш запит",
    ],
    "en": [
        "Hello! Working on your request",
        "Hello! Our team is reviewing your request",
        "Hello! Your request has been received and is being processed",
        "Hello! We’re checking this now",
        "Hello! Thanks for your message — we’re on it",
        "Hello! Our team is processing your request",
        "Hello! Your request is in progress",
        "Hello! Thanks for reaching out — we’re on it",
        "Hello! We’ve got your request and are working on it"
    ],
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
        "наберіть мене", "подзвоніть", "можете подзвонити", "позвоніть",
        "передзвоніть", "зателефонуйте", "наберіть", "перезвоніть",
        "пазвоніть",
    ],
    "en": [
        "call me", "please call", "can you call",
        "give me a call", "call please", "call me please"
    ]
}
CALL_REPLY = {
    "ru": "Здравствуйте! Пару минут, пожалуйста, и мы Вас наберём",
    "uk": "Вітаємо! Декілька хвилин, будь ласка, і ми Вам зателефонуємо",
    "en": "Hello! Please give us a couple of minutes and we will call you"
}

CLOSE_KEYWORDS = [
    "done", "all done", "have a good day", "stay safe", "ready", "shift started", "safe trip", "fixed your violation",
    "fixed", "added", "marked", "added new shipping", "added the load", "log been set",
    "have a nice day", "have a good rest", "updated pickup time", "started the shift", "have a nice trip",
    "have a good one", "have a great day", "fixed your log", "made a split", "made a cycle reset",
    "violations fixed", "fixed violations", "time added", "BOL added", "added some time", "have a safe trip", 
    "shift opened", "all the best", "all fixed", "all set", "log fixed", "shift available", "split activated", "request completed",
    "co-driver drop", "shift reopened", "new shift opened", "info added", "have a nice rest", "have a great rest",
    "log fixed", "logbook updated", "logbook fixed", "started shift", "information added", "added information", 
    "added info", "added time", "added break", "break added", "PTI added", "added PTI", "stay and drive safe",
    "готово", "хорошей дороги", "безопасной дороги", "хорошего дня", "всего наилучшего", "хорошего отдыха",
    "всего доброго", "новая смена доступна", "новая смена открыта", "смена открыта", "открыли смену", 
    "брейк добавлен", "смена доступна", "поправили", "сделали сплит", "активировали сплит", 
    "сделали вам сплит", "запрос выполнен", "сделали сброс цикла", "добавили груз в логбук", "сделали брейк",
    "все исправили", "все поправили", "все готово", "хорошего дня и безопасной дороги", "хорошей и безопасной дороги",
    "добавили время", "добавили времени", "хорошего вам дня", "удачного вам дня", "начали смену", "смена началась",
    "добавили брейк", "добавили PTI", "PTI добавлен", "ПТИ добавлен", "ПТИ добавили", "сделали ПТИ",
    "гарного дня", "безпечної дороги", "відкрили зміну", "гарної дороги", "вдалої дороги", 
    "всього найкращого", "зміна відкрита", "зробили", "додали", "зміна доступна", "сделали сплит", "безпечної вам дороги",
    "активували спліт", "зробили спліт", "зробили вам спліт", "запит виконано", "гарного відпочинку", "всього доброго",
    "зробили скидання циклу", "нова зміна відкрита", "додали часу", "усе готово", "час додано", "вдалої та безпечної дороги",
    "гарного дня та безпечної дороги", "гарної та безпечної дороги", "вдалого вам дня", "гарного вам дня",
    "зробили рестарт циклу", "зробили вам рестарт циклу", "рестарт циклу зроблено", "цикл оновлено", "удачного пути",
    "PTI додано", "додали PTI", "ПТІ додано", "додали ПТІ", "ПТИ додано", "додали ПТИ", "счастливого пути", "логбук готов"
]

THRESHOLD = 0.35 
open_tasks = {}
pending_media_checks = {}
recently_closed = {}
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

def get_reply(lang: str, reply_dict: dict) -> str:
    texts = reply_dict.get(lang) or reply_dict.get("en")
    
    if isinstance(texts, list):
        text = random.choice(texts)
    else:
        text = texts
    
    emoji = random.choice(EMOJIS)
    return f"{text} {emoji}"

bot = Bot(token=TG_TOKEN)
dp = Dispatcher()

async def send_to_discord(text: str):
    global session
    if session is None or session.closed:
        connector = aiohttp.TCPConnector(limit=10)
        session = aiohttp.ClientSession(connector=connector)
    
    for attempt in range(3):  
        try:
            async with session.post(DISCORD_WEBHOOK, json={"content": text}, timeout=10) as response:

                if response.status == 204:
                    return

                if response.status == 429:
                    retry_after = 10

                    try:
                        data = await response.json()
                        retry_after = data.get("retry_after", 10)
                    except:
                        pass

                    print(f"[{datetime.utcnow()}] Rate limited. Sleep {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue

                if response.status >= 400:
                    error_text = await response.text()
                    print(f"[{datetime.utcnow()}] Discord Error {response.status}: {error_text}")
                    return

        except Exception as e:
            print(f"[{datetime.utcnow()}] Connection Error: {e}")
            await asyncio.sleep(10)

@dp.callback_query()
async def handle_task_buttons(callback: CallbackQuery):
    if callback.from_user.id not in ADMIN_IDS:
        return

    data = callback.data

    if data.startswith("close_"):
        chat_id = int(data.replace("close_", ""))
        task = open_tasks.get(chat_id)
        if not task:
            await callback.answer("Task already closed", show_alert=True)
            return

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Yes", callback_data=f"do_close_{chat_id}"),
                InlineKeyboardButton(text="❌ No", callback_data="cancel_close")
            ]
        ])
        await callback.message.edit_text(
            f"Are you sure you want to close task '{task['title']}'?",
            reply_markup=keyboard
        )
        await callback.answer()
        return

    if data.startswith("do_close_"):
        chat_id = int(data.replace("do_close_", ""))
        task = open_tasks.get(chat_id)
        if not task:
            await callback.answer("Task already closed", show_alert=True)
            return

        delta = datetime.utcnow() - task["opened_at"]

        recently_closed[chat_id] = datetime.utcnow()
        
        del open_tasks[chat_id]
        if chat_id in pending_media_checks:
            pending_media_checks.pop(chat_id, None)

        if open_tasks:
            keyboard = InlineKeyboardMarkup(inline_keyboard=[])
            now = datetime.utcnow()
            for cid, t in open_tasks.items():
                delta = now - t["opened_at"]
                mins = int(delta.total_seconds() // 60)
                keyboard.inline_keyboard.append(
                    [InlineKeyboardButton(
                        text=f"{t['title']} — {mins} min",
                        callback_data=f"close_{cid}"
                    )]
                )
            await callback.message.edit_text("Select the task you want to close. Open tasks:", reply_markup=keyboard)
        else:
            await callback.message.edit_text("All tasks are closed ✅")
        return

    if data == "cancel_close":
        if open_tasks:
            keyboard = InlineKeyboardMarkup(inline_keyboard=[])
            now = datetime.utcnow()
            for cid, t in open_tasks.items():
                delta = now - t["opened_at"]
                mins = int(delta.total_seconds() // 60)
                keyboard.inline_keyboard.append(
                    [InlineKeyboardButton(
                        text=f"{t['title']} — {mins} min",
                        callback_data=f"close_{cid}"
                    )]
                )
            await callback.message.edit_text("Select the task you want to close. Open tasks:", reply_markup=keyboard)
        else:
            await callback.message.edit_text("There are no open tasks")
        await callback.answer()

async def handle_admin_command(msg: types.Message):
    text = (msg.text or "").strip()

    if text == "/tasks":
        if not open_tasks:
            await msg.answer("There are no open tasks")
            return

        keyboard = InlineKeyboardMarkup(inline_keyboard=[])

        now = datetime.utcnow()

        for chat_id, task in open_tasks.items():
            delta = now - task["opened_at"]
            minutes = int(delta.total_seconds() // 60)

            button_text = f"{task['title']} — {minutes} min"
            keyboard.inline_keyboard.append(
                [InlineKeyboardButton(
                    text=button_text,
                    callback_data=f"close_{chat_id}"
                )]
            )

        await msg.answer("Open tasks:", reply_markup=keyboard)
        return

    await msg.answer(
        "⚙️ Commands:\n"
        "/tasks — task list\n"
    )

@dp.message()
async def on_message(msg: types.Message):
    if msg.from_user.is_bot:
        return
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
    light_media = bool(msg.sticker or msg.animation or msg.poll or msg.contact or msg.location or msg.venue)

    if msg.from_user.id in ADMIN_IDS:
        normalized_text = re.sub(r"[^a-zA-Zа-яА-ЯёЁіІїЇєЄ\s]", "", text).lower()
        
        for phrase in CLOSE_KEYWORDS:
            phrase_clean = re.sub(r"[^a-zA-Zа-яА-ЯёЁіІїЇєЄ\s]", "", phrase).lower()

            if len(phrase_clean.split()) == 1:
                pattern = r"\b" + re.escape(phrase_clean) + r"\b"
                if re.search(pattern, normalized_text):
                    recently_closed[chat_id] = datetime.utcnow()

                    if chat_id in open_tasks:
                        del open_tasks[chat_id]

                    if chat_id in pending_media_checks:
                        pending_media_checks.pop(chat_id, None)
                    return
            else:
                if phrase_clean in normalized_text:
                    recently_closed[chat_id] = datetime.utcnow()

                    if chat_id in open_tasks:
                        del open_tasks[chat_id]

                    if chat_id in pending_media_checks:
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
        now = datetime.utcnow()

        if chat_id in recently_closed:
            closed_at = recently_closed[chat_id]

            if now - closed_at < timedelta(minutes=30):
                open_tasks[chat_id] = {
                    "title": chat_title,
                    "opened_at": now,
                    "notifications_sent": [],
                }
                await discord_queue.put(f"**{chat_title} requested assistance again**")
                if chat_id in pending_media_checks:
                    pending_media_checks.pop(chat_id, None)
                return
            else:
                del recently_closed[chat_id]

        open_tasks[chat_id] = {
            "title": chat_title,
            "opened_at": now,
            "notifications_sent": [],
        }

        if is_call:
            await msg.answer(get_reply(lang, CALL_REPLY))
        else:
            await msg.answer(get_reply(lang, AUTO_REPLY))
        
        await discord_queue.put(msg.chat.title)

        if chat_id in pending_media_checks:
            pending_media_checks.pop(chat_id, None)
        return

    if is_media:
        if chat_id not in pending_media_checks:
            async def media_check():
                try:
                    await asyncio.sleep(30)
                    now = datetime.utcnow()

                    if light_media and chat_id in recently_closed:
                        closed_at = recently_closed[chat_id]
                        if now - closed_at < timedelta(minutes=30):
                            return
                        else:
                            del recently_closed[chat_id]

                    if chat_id not in open_tasks:
                        open_tasks[chat_id] = {
                            "title": chat_title,
                            "opened_at": datetime.utcnow(),
                            "notifications_sent": [],
                        }
                        await discord_queue.put(f"**{chat_title} needs an answer**")
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
                    await discord_queue.put(f"**{group_name} task open for more than {minutes} minutes**")
                    task["notifications_sent"].append(minutes)

            if now - task["opened_at"] > timedelta(hours=1):
                del open_tasks[chat_id]

        await asyncio.sleep(120)

async def discord_worker():
    print("DISCORD WORKER STARTED")
    while True:
        text = await discord_queue.get()
        print(f"SEND: {text}")
        print(f"QUEUE SIZE: {discord_queue.qsize()}")
        try:
            await send_to_discord(text)
        except Exception as e:
            print(f"[{datetime.utcnow()}] Worker error: {e}")

        await asyncio.sleep(10)

async def main():
    global session
    connector = aiohttp.TCPConnector(limit=10) 
    session = aiohttp.ClientSession(connector=connector)

    monitor_task = asyncio.create_task(monitor_tasks())
    print(f"[{datetime.utcnow()}] Bot started and monitoring active")
    worker_task = asyncio.create_task(discord_worker())

    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
    except Exception as e:
        print(f"[{datetime.utcnow()}] Critical polling error: {e}")
    finally:
        monitor_task.cancel()
        worker_task.cancel()
        await session.close()
        print(f"[{datetime.utcnow()}] Bot stopped, session closed")

if __name__ == "__main__":
    asyncio.run(main())