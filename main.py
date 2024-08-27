import yt_dlp
from pydub import AudioSegment
import subprocess
import json
import os
import re
import trafilatura
import uuid
import requests
from litellm import completion
from duckduckgo_search import AsyncDDGS
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CommandHandler, MessageHandler, CallbackQueryHandler, filters, ApplicationBuilder
from youtube_transcript_api import YouTubeTranscriptApi

# 從環境變數中取得 OpenAI API Key
openai_api_key = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY")
telegram_token = os.environ.get("TELEGRAM_TOKEN", "xxx")
model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
lang = os.environ.get("TS_LANG", "繁體中文")
ddg_region = os.environ.get("DDG_REGION", "wt-wt")
chunk_size = int(os.environ.get("CHUNK_SIZE", 2100))
allowed_users = os.environ.get("ALLOWED_USERS", "")
use_audio_fallback = int(os.environ.get("USE_AUDIO_FALLBACK", "0"))

def split_user_input(text):
    paragraphs = text.split('\n')
    paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]
    return paragraphs

def scrape_text_from_url(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded, include_formatting=True)
        if text is None:
            return []
        text_chunks = text.split("\n")
        article_content = [text for text in text_chunks if text]
        return article_content
    except Exception as e:
        print(f"Error: {e}")

async def search_results(keywords):
    print(keywords, ddg_region)
    results = await AsyncDDGS().text(keywords, region=ddg_region, safesearch='off', max_results=6)
    return results

def summarize(text_array):
    def create_chunks(paragraphs):
        chunks = []
        chunk = ''
        for paragraph in paragraphs:
            if len(chunk) + len(paragraph) < chunk_size:
                chunk += paragraph + ' '
            else:
                chunks.append(chunk.strip())
                chunk = paragraph + ' '
        if chunk:
            chunks.append(chunk.strip())
        return chunks

    try:
        text_chunks = create_chunks(text_array)
        text_chunks = [chunk for chunk in text_chunks if chunk]

        summaries = []
        system_messages = [
            {"role": "system", "content": "將以下原文總結為四個部分：總結 (Overall Summary)。觀點 (Viewpoints)。摘要 (Abstract)： 創建6到10個帶有適當表情符號的重點摘要。關鍵字 (Key Words)。請確保每個部分只生成一次，且內容不重複。確保生成的文字都是{lang}為主"}
        ]

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(call_gpt_api, f"總結 the following text:\n{chunk}", system_messages) for chunk in text_chunks]
            summaries = [future.result() for future in tqdm(futures, total=len(text_chunks), desc="Summarizing")]

        final_summary = {
            "overall_summary": "",
            "viewpoints": "",
            "abstract": "",
            "keywords": ""
        }
        for summary in summaries:
            if '總結 (Overall Summary)' in summary and not final_summary["overall_summary"]:
                final_summary["overall_summary"] = summary.split('觀點 (Viewpoints)')[0].strip()
            if '觀點 (Viewpoints)' in summary and not final_summary["viewpoints"]:
                content = summary.split('摘要 (Abstract)')[0].split('觀點 (Viewpoints)')[1].strip()
                final_summary["viewpoints"] = content
            if '摘要 (Abstract)' in summary and not final_summary["abstract"]:
                content = summary.split('關鍵字 (Key Words)')[0].split('摘要 (Abstract)')[1].strip()
                final_summary["abstract"] = content
            if '關鍵字 (Key Words)' in summary and not final_summary["keywords"]:
                content = summary.split('關鍵字 (Key Words)')[1].strip()
                final_summary["keywords"] = content

        output = "\n\n".join([
            f"  歡迎使用 Oli 家 小濃縮機器人 (Summary) \n{final_summary['overall_summary']}",
            f" **觀點 (Viewpoints)**\n{final_summary['viewpoints']}",
            f" **摘要 (Abstract)**\n{final_summary['abstract']}",
            f" **關鍵字 (Key Words)**\n{final_summary['keywords']}"
        ])
        return output
    except Exception as e:
        print(f"Error: {e}")
        return "Unknown error! Please contact the owner. ok@vip.david888.com"

def extract_youtube_transcript(youtube_url):
    try:
        video_id_match = re.search(r"(?<=v=)[^&]+|(?<=youtu.be/)[^?|\n]+", youtube_url)
        video_id = video_id_match.group(0) if video_id_match else None
        if video_id is None:
            return "no transcript"
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        available_languages = [transcript.language_code for transcript in transcript_list]
        transcript = transcript_list.find_transcript(available_languages)
        transcript_text = ' '.join([item['text'] for item in transcript.fetch()])
        return transcript_text
    except Exception as e:
        print(f"Error: {e}")
        return "no transcript"

def retrieve_yt_transcript_from_url(youtube_url):
    try:
        output = extract_youtube_transcript(youtube_url)
        if output == 'no transcript':
            if use_audio_fallback:
                raise ValueError("There's no valid transcript in this video. Falling back to audio transcription.")
            else:
                return ["該影片沒有可用的字幕。"]

        output_sentences = output.split(' ')
        output_chunks = []
        current_chunk = ""

        for sentence in output_sentences:
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += sentence + ' '
            else:
                output_chunks.append(current_chunk.strip())
                current_chunk = sentence + ' '

        if current_chunk:
            output_chunks.append(current_chunk.strip())
        return output_chunks

    except Exception as e:
        print(f"Error: {e}")
        if not use_audio_fallback:
            return ["無法獲取字幕，且音頻轉換功能未啟用。"]


        # 以下是音頻轉換的代碼，只有在 use_audio_fallback 為 True 時才執行
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'/tmp/{str(uuid.uuid4())}.%(ext)s',
            'ffmpeg_location': '/usr/bin/ffmpeg',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'ffprobe_location': '/usr/bin/ffprobe'
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            output_path = ydl.prepare_filename(info)

        output_path = output_path.replace(os.path.splitext(output_path)[1], ".mp3")
        audio_file = AudioSegment.from_file(output_path)
 

        chunk_size = 100 * 1000  # 100 秒
        chunks = [audio_file[i:i+chunk_size] for i in range(0, len(audio_file), chunk_size)]

        transcript = ""
        for i, chunk in enumerate(chunks):
            temp_file_path = f"/tmp/{str(uuid.uuid4())}.wav"
            chunk.export(temp_file_path, format="wav")

            curl_command = [
                "curl",
                "https://api.openai.com/v1/audio/transcriptions",
                "-H", f"Authorization: Bearer {openai_api_key}",
                "-H", "Content-Type: multipart/form-data",
                "-F", f"file=@{temp_file_path}",
                "-F", "model=whisper-1"
            ]

            result = subprocess.run(curl_command, capture_output=True, text=True)

            try:
                response_json = json.loads(result.stdout)
                transcript += response_json["text"]
            except KeyError as e:
                print("KeyError:", e)
                print("Response JSON:", response_json)
            except json.JSONDecodeError:
                print("Failed to decode JSON:", result.stdout)

        output_sentences = transcript.split(' ')
        output_chunks = []
        current_chunk = ""

        for sentence in output_sentences:
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += sentence + ' '
            else:
                output_chunks.append(current_chunk.strip())
                current_chunk = sentence + ' '

        if current_chunk:
            output_chunks.append(current_chunk.strip())

        return output_chunks

def call_gpt_api(prompt, additional_messages=[]):
    try:
        response = completion(
            model=model,
            messages=additional_messages + [
                {"role": "user", "content": prompt}
            ],
        )
        message = response.choices[0].message.content.strip()
        return message
    except Exception as e:
        print(f"Error: {e}")
        return ""

async def handle_start(update, context):
    return await handle('start', update, context)

async def handle_help(update, context):
    return await handle('help', update, context)

async def handle_summarize(update, context):
    return await handle('summarize', update, context)

async def handle_file(update, context):
    return await handle('file', update, context)

async def handle_button_click(update, context):
    return await handle('button_click', update, context)


async def handle_yt2audio(update, context):
    chat_id = update.effective_chat.id
    user_input = update.message.text.split()

    if len(user_input) < 2:  # 檢查是否有提供 URL
        await context.bot.send_message(chat_id=chat_id, text="請提供一個 YouTube 影片的 URL。例如：/yt2audio https://www.youtube.com/watch?v=OrUQJg_vFKE")
        return

    url = user_input[1]  # 取得 YouTube URL

    try:
        # 使用 yt-dlp 下載音頻
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'/tmp/{str(uuid.uuid4())}.%(ext)s',  # 直接使用這個模板來生成文件名
            'ffmpeg_location': '/usr/bin/ffmpeg',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'ffprobe_location': '/usr/bin/ffprobe'
        }


        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # 不再使用 replace，直接使用下載後的文件
        output_path = ydl_opts['outtmpl']  # 這裡是帶有 "%(ext)s" 的模板

        # 如果你確定已經下載為 .mp3，可以直接用文件路徑
        output_path = output_path.replace("%(ext)s", "mp3")  # 如果你想保留這行也可以，確保文件是 mp3 格式

        audio_file = AudioSegment.from_file(output_path)        
 


            
        # 傳送音頻檔案給 Telegram user
        with open(output_path, 'rb') as audio:
            await context.bot.send_audio(chat_id=chat_id, audio=audio)

        os.remove(output_path)  # 刪除臨時檔案       
  

    except Exception as e:
        print(f"Error: {e}")
        await context.bot.send_message(chat_id=chat_id, text="下載或傳送音頻失敗。請檢查輸入的 YouTube URL 是否正確。")
        


async def handle_yt2text(update, context):
    chat_id = update.effective_chat.id
    user_input = update.message.text.split()

    if len(user_input) < 2:
        await context.bot.send_message(chat_id=chat_id, text="請提供一個 YouTube 影片的 URL。例如：/yt2text https://www.youtube.com/watch?v=OrUQJg_vFKE")
        return

    url = user_input[1]

    try:
        output_chunks = retrieve_yt_transcript_from_url(url)

        if len(output_chunks) == 1 and (output_chunks[0] == "該影片沒有可用的字幕。" or output_chunks[0] == "無法獲取字幕，且音頻轉換功能未啟用。"):
            await context.bot.send_message(chat_id=chat_id, text=output_chunks[0])
            return

        # 處理正常情況的代碼
        temp_file_path = f"/tmp/{str(uuid.uuid4())}.txt"
        with open(temp_file_path, 'w', encoding='utf-8') as file:
            for chunk in output_chunks:
                file.write(chunk + "\n")

        with open(temp_file_path, 'rb') as txt_file:
            await context.bot.send_document(chat_id=chat_id, document=txt_file, filename="transcript.txt")

        os.remove(temp_file_path)  # 刪除臨時檔案

    except Exception as e:
        print(f"Error: {e}")
        await context.bot.send_message(chat_id=chat_id, text="下載或轉換文本失敗。請檢查輸入的 YouTube URL 是否正確。")

        
def process_user_input(user_input):
    youtube_pattern = re.compile(r"https?://(www\.|m\.)?(youtube\.com|youtu\.be)/")
    url_pattern = re.compile(r"https?://")

    if youtube_pattern.match(user_input):
        text_array = retrieve_yt_transcript_from_url(user_input)
    elif url_pattern.match(user_input):
        text_array = scrape_text_from_url(user_input)
    else:
        text_array = split_user_input(user_input)

    return text_array

def get_inline_keyboard_buttons():
    keyboard = [
        [InlineKeyboardButton("Explore Similar", callback_data="explore_similar")],
        [InlineKeyboardButton("Why It Matters", callback_data="why_it_matters")],
    ]
    return InlineKeyboardMarkup(keyboard)

def clear_old_commands(telegram_token):
    url = f"https://api.telegram.org/bot{telegram_token}/deleteMyCommands"
    
    scopes = ["default", "all_private_chats", "all_group_chats", "all_chat_administrators"]
    
    for scope in scopes:
        data = {"scope": {"type": scope}}
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            print(f"Old commands cleared successfully for scope: {scope}")
        else:
            print(f"Failed to clear old commands for scope {scope}: {response.text}")

def set_my_commands(telegram_token):
    clear_old_commands(telegram_token)  # 清除舊的命令
    url = f"https://api.telegram.org/bot{telegram_token}/setMyCommands"
    commands = [
        {"command": "start", "description": "確認機器人是否在線"},
        {"command": "help", "description": "顯示此幫助訊息"},
        {"command": "yt2audio", "description": "下載 YouTube 音頻"},
        {"command": "yt2text", "description": "將 YouTube 影片轉成文字"},
    ]
    data = {"commands": commands}
    response = requests.post(url, json=data)

    if response.status_code == 200:
        print("Commands set successfully.")
    else:
        print(f"Failed to set commands: {response.text}")
        
async def handle(action, update, context):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    if allowed_users and str(user_id) not in allowed_users.split(','):
        await context.bot.send_message(chat_id=chat_id, text="Sorry, you are not authorized to use this bot.")
        return

    if action == 'start':
        await context.bot.send_message(chat_id=chat_id, text="Welcome! I'm here to help you summarize text and YouTube videos.")
    elif action == 'help':
        help_text = """
        Here are the available commands:
        /start - Start the bot
        /help - Show this help message
        /yt2audio <YouTube URL> - Download YouTube audio
        /yt2text <YouTube URL> - Convert YouTube video to text
        
        You can also send me any text or URL to summarize.
        """
        await context.bot.send_message(chat_id=chat_id, text=help_text)
    elif action == 'summarize':
        user_input = update.message.text
        text_array = process_user_input(user_input)
        if text_array:
            summary = summarize(text_array)
            await context.bot.send_message(chat_id=chat_id, text=summary, reply_markup=get_inline_keyboard_buttons())
        else:
            await context.bot.send_message(chat_id=chat_id, text="Sorry, I couldn't process your input. Please try again.")
    elif action == 'file':
        file = await update.message.document.get_file()
        file_path = f"/tmp/{file.file_id}.pdf"
        await file.download_to_drive(file_path)
        
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        os.remove(file_path)
        
        text_array = text.split("\n")
        summary = summarize(text_array)
        await context.bot.send_message(chat_id=chat_id, text=summary, reply_markup=get_inline_keyboard_buttons())
    elif action == 'button_click':
        query = update.callback_query
        await query.answer()
        
        if query.data == 'explore_similar':
            await context.bot.send_message(chat_id=chat_id, text="Here are some similar topics...")
        elif query.data == 'why_it_matters':
            await context.bot.send_message(chat_id=chat_id, text="This topic matters because...")



def main():
    try:
        application = ApplicationBuilder().token(telegram_token).build()
        start_handler = CommandHandler('start', handle_start)
        help_handler = CommandHandler('help', handle_help)
        yt2audio_handler = CommandHandler('yt2audio', handle_yt2audio)
        yt2text_handler = CommandHandler('yt2text', handle_yt2text)
        set_my_commands(telegram_token)
        summarize_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, handle_summarize)
        file_handler = MessageHandler(filters.Document.PDF, handle_file)
        button_click_handler = CallbackQueryHandler(handle_button_click)
        application.add_handler(file_handler)
        application.add_handler(start_handler)
        application.add_handler(help_handler)
        application.add_handler(yt2audio_handler)
        application.add_handler(yt2text_handler)
        application.add_handler(summarize_handler)
        application.add_handler(button_click_handler)
        application.run_polling()
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()

