def count(state: dict) -> dict:
    count = state.get("count", 0)
    new_count = count + 1
    # print(f"Flow has run {new_count} time(s)")
    return {
        "count": new_count
    }

def stop_count(state: dict) -> str:
    count = state.get("count", 0)

    # print(f"Checking count ={count} should be stopped")

    if count >= 10:
        return "True"
    else:
        return "False"