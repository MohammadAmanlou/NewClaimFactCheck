import requests
import json
import time

def verify_keys(key_file):
    with open(key_file, 'r', encoding='utf-8') as f:
        keys = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(keys)} keys from '{key_file}'. Checking status...\n")

    suspended = []
    active = []
    for i, key in enumerate(keys):
        line_num = i + 1
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
        try:
            res = requests.get(url, timeout=10, verify=False)
            if res.status_code == 200:
                print(f"[{line_num}] SUCCESS (Active)")
                active.append((line_num, key))
            elif res.status_code == 403:
                print(f"[{line_num}] SUSPENDED (403 Forbidden)")
                suspended.append((line_num, key))
            elif res.status_code == 400:
                print(f"[{line_num}] INVALID (400 Bad Request)")
                suspended.append((line_num, key))
            else:
                print(f"[{line_num}] OTHER: {res.status_code} - {res.text.strip()}")
        except Exception as e:
            print(f"[{line_num}] ERROR: {str(e)}")
            
        time.sleep(0.1) # Be nice to the API

    print("\n" + "="*40)
    print(f"RESULTS:")
    print(f"Active Keys: {len(active)}")
    print(f"Suspended/Invalid Keys: {len(suspended)}")
    print("="*40)

    if suspended:
        print("\nList of Suspended/Invalid lines:")
        for line, k in suspended:
            print(f"Line {line}")
            
    # Optionally, we could create a new file with only the active keys
    if active:
        with open("google_api_list_active_only.txt", "w", encoding="utf-8") as f:
            for _, k in active:
                f.write(k + "\n")
        print("\n[i] Exported all known active keys into 'google_api_list_active_only.txt'")
        
if __name__ == "__main__":
    verify_keys("google_api_list.txt")