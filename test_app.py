from fastapi import FastAPI

app = FastAPI(title="Nirvana Backend Test")

@app.get("/")
def read_root():
    return {"message": "Nirvana Backend is running!"}

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "nirvana-backend"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
