import logging

import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name('creds/happyai_google_sheets_creds.json', scope)
client = gspread.authorize(creds)
sheet = client.open("HappyAI Trainees. RAG task. March 2025").get_worksheet_by_id(1774489503)

logging.info('Успешно подключились к гугл таблице')