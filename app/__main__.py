import uvicorn

from . import config

uvicorn.run("app.main:app", host=config.HOST, port=config.PORT)
