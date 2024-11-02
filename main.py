import asyncio
import os
import re
import trafilatura
from litellm import completion
from duckduckgo_search import AsyncDDGS
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CommandHandler, MessageHandler, CallbackQueryHandler, filters, ApplicationBuilder
from youtube_transcript_api import YouTubeTranscriptApi
from newspaper import Article
from goose3 import Goose

telegram_token = os.environ.get("TELEGRAM_TOKEN", "xxx")
model = os.environ.get("LLM_MODEL", "gpt-3.5-turbo-16k")
lang = os.environ.get("TS_LANG", "Taiwanese Mandarin")
ddg_region = os.environ.get("DDG_REGION", "wt-wt")
chunk_size = int(os.environ.get("CHUNK_SIZE", 10000))
allowed_users = os.environ.get("ALLOWED_USERS", "")

def split_user_input(text):
    # Split the input text into paragraphs
    paragraphs = text.split('\n')

    # Remove empty paragraphs and trim whitespace
    paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]

    return paragraphs

def scrape_text_from_url(url):
    """
    Scrape the content from the URL
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded, include_formatting=True)
        if text is None:
            raise ValueError("Trafilatura returned None")
    except Exception as e:
        print(f"Trafilatura failed: {e}")
        try:
            print("Trying newspaper3k...")
            article = Article(url)
            article.download()
            article.parse()
            text = article.text
            if not text:
                raise ValueError("Newspaper3k returned empty text")
        except Exception as e:
            print(f"Newspaper3k failed: {e}")
            try:
                print("Trying Goose3...")
                g = Goose()
                article = g.extract(url=url)
                text = article.cleaned_text
                if not text:
                    raise ValueError("Goose3 returned empty text")
            except Exception as e:
                print(f"Goose3 failed: {e}")
                return []

    text_chunks = text.split("\n")
    article_content = [text for text in text_chunks if text]
    return article_content

async def search_results(keywords):
    print(keywords, ddg_region)
    results = await AsyncDDGS().text(keywords, region=ddg_region, safesearch='off', max_results=3)
    return results

def summarize(text_array):
    """
    Summarize the text using GPT API
    """

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
        text_chunks = [chunk for chunk in text_chunks if chunk] # Remove empty chunks

        # Call the GPT API in parallel to summarize the text chunks
        summaries = []
        system_messages = [
            {"role": "system", "content": "You are an expert in creating summaries that capture the main points and key details."},
            {"role": "system", "content": f"You will show the bulleted list content without translate any technical terms."},
            {"role": "system", "content": f"You will print all the content in {lang}."},
        ]
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(call_gpt_api, f"Summary keypoints for the following text:\n{chunk}", system_messages) for chunk in text_chunks]
            for future in tqdm(futures, total=len(text_chunks), desc="Summarizing"):
                summaries.append(future.result())

        if len(summaries) <= 5:
            summary = ' '.join(summaries)
            with tqdm(total=1, desc="Final summarization") as progress_bar:
                final_summary = call_gpt_api(f"Create a bulleted list using {lang} to show the key points of the following text:\n{summary}", system_messages)
                progress_bar.update(1)
            return final_summary
        else:
            return summarize(summaries)
    except Exception as e:
        print(f"Error: {e}")
        return "Unknown error! Please contact the developer."

def extract_youtube_transcript(youtube_url):
    try:
        video_id_match = re.search(r"(?<=v=)[^&]+|(?<=youtu.be/)[^?|\n]+", youtube_url)
        video_id = video_id_match.group(0) if video_id_match else None
        if video_id is None:
            return "no transcript"
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(['en', 'ja', 'ko', 'de', 'fr', 'ru', 'it', 'es', 'pl', 'uk', 'nl', 'zh-TW', 'zh-CN', 'zh-Hant', 'zh-Hans'])
        transcript_text = ' '.join([item['text'] for item in transcript.fetch()])
        return transcript_text
    except Exception as e:
        print(f"Error: {e}")
        return "no transcript"

def retrieve_yt_transcript_from_url(youtube_url):
    output = extract_youtube_transcript(youtube_url)
    if output == 'no transcript':
        raise ValueError("There's no valid transcript in this video.")
    # Split output into an array based on the end of the sentence (like a dot),
    # but each chunk should be smaller than chunk_size
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

def call_gpt_api(prompt, additional_messages=[]):
    """
    Call GPT API
    """
    try:
        response = completion(
        # response = openai.ChatCompletion.create(
            model=model,
            messages=additional_messages+[
                {"role": "user", "content": prompt}
            ],

        )
        message = response.choices[0].message.content.strip()
        return message
    except Exception as e:
        print(f"Error: {e}")
        return ""

def handle_start(update, context):
    return handle('start', update, context)

def handle_help(update, context):
    return handle('help', update, context)

def handle_summarize(update, context):
    return handle('summarize', update, context)

def handle_file(update, context):
    return handle('file', update, context)

def handle_button_click(update, context):
    return handle('button_click', update, context)

async def handle(command, update, context):
    chat_id = update.effective_chat.id
    print("chat_id=", chat_id)

    if allowed_users:
        user_ids = allowed_users.split(',')
        if str(chat_id) not in user_ids:
            print(chat_id, "is not allowed.")
            await context.bot.send_message(chat_id=chat_id, text="You have no permission to use this bot.")
            return

    try:
        if command == 'start':
            await context.bot.send_message(chat_id=chat_id, text="I can summarize text, URLs, PDFs and YouTube video for you.")
        elif command == 'help':
            await context.bot.send_message(chat_id=chat_id, text="Report bugs here 👉 https://github.com/tpai/summary-gpt-bot/issues", disable_web_page_preview=True)
        elif command == 'summarize':
            user_input = update.message.text
            print("user_input=", user_input)

            text_array = process_user_input(user_input)
            print(text_array)

            if not text_array:
                raise ValueError("No content found to summarize.")

            await context.bot.send_chat_action(chat_id=chat_id, action="TYPING")
            summary = summarize(text_array)
            await context.bot.send_message(chat_id=chat_id, text=f"{summary}", reply_to_message_id=update.message.message_id, reply_markup=get_inline_keyboard_buttons())
        elif command == 'file':
            file_path = f"{update.message.document.file_unique_id}.pdf"
            print("file_path=", file_path)

            file = await context.bot.get_file(update.message.document)
            await file.download_to_drive(file_path)

            text_array = []
            reader = PdfReader(file_path)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                text_array.append(text)

            await context.bot.send_chat_action(chat_id=chat_id, action="TYPING")
            summary = summarize(text_array)
            await context.bot.send_message(chat_id=chat_id, text=f"{summary}", reply_to_message_id=update.message.message_id, reply_markup=get_inline_keyboard_buttons())

            # remove temp file after sending message
            os.remove(file_path)
        elif command == 'button_click':
            original_message_text = update.callback_query.message.text
            await context.bot.send_chat_action(chat_id=chat_id, action="TYPING")

            if update.callback_query.data == "explore_similar":
                keywords = call_gpt_api(f"{original_message_text}\nBased on the content above, give me the top 5 important keywords with commas.", [
                    {"role": "system", "content": f"You will print keywords only."}
                ])

                tasks = [search_results(keywords)]
                results = await asyncio.gather(*tasks)
                print(results)

                links = ''
                for r in results[0]:
                    links += f"{r['title']}\n{r['href']}\n"

                await context.bot.send_message(chat_id=chat_id, text=links, reply_to_message_id=update.callback_query.message.message_id, disable_web_page_preview=True)

            if update.callback_query.data == "why_it_matters":
                result = call_gpt_api(f"{original_message_text}\nBased on the content above, tell me why it matters as an expert.", [
                    {"role": "system", "content": f"You will show the result in {lang}."}
                ])
                await context.bot.send_message(chat_id=chat_id, text=result, reply_to_message_id=update.callback_query.message.message_id)
    except Exception as e:
        print(f"Error: {e}")
        await context.bot.send_message(chat_id=chat_id, text=str(e))


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

def main():
    try:
        application = ApplicationBuilder().token(telegram_token).build()
        start_handler = CommandHandler('start', handle_start)
        help_handler = CommandHandler('help', handle_help)
        summarize_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, handle_summarize)
        file_handler = MessageHandler(filters.Document.PDF, handle_file)
        button_click_handler = CallbackQueryHandler(handle_button_click)
        application.add_handler(file_handler)
        application.add_handler(start_handler)
        application.add_handler(help_handler)
        application.add_handler(summarize_handler)
        application.add_handler(button_click_handler)
        application.run_polling()
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()
