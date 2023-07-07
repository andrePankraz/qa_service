"""
This file was created by ]init[ AG 2022.

Module for QA Service.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import os

logger = logging.getLogger(__name__)


app = FastAPI(root_path=os.getenv("BASE_PATH", "/"))

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:8080",
    # Add more allowed origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    logger.info("Startup...")
    app.mount("/", StaticFiles(directory="src/qa_service/resources/html", html=True), name="static")


@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down...")


# Following ordering is important for overlapping path matches...
