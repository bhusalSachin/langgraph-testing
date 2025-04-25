from src.count.graph import app as count_app

if __name__ == "__main__":
    state = count_app.invoke(
        {
            "count": 0
        },
        config={
            "recursion_limit": 150,
            "thread_id": f"42" 
        },
    )