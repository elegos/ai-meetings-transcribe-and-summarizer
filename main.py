from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from routers.process import process_router

app = FastAPI()
app.include_router(process_router)

@app.get('/', response_class=HTMLResponse)
async def index():
    with open('static/index.html', 'r') as f:
        content = f.read()

    return content

app.mount('/', StaticFiles(directory='static'), name='static')
