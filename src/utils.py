import time
import requests
import socket
import platform
import uuid
import psutil
import datetime

# Function to stream the response
def stream_response(response, delay=0.01):
    for res in response:
        yield res
        time.sleep(delay)

# Function to remove Lucene special characters
def remove_lucene_chars_cust(text: str) -> str:
    """Remove Lucene special characters"""
    special_chars = [
        "+",
        "-",
        "&",
        "|",
        "!",
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
        "^",
        '"',
        "~",
        "*",
        "?",
        ":",
        "\\",
        "/"
    ]

    for char in special_chars:
        if char in text:
            text = text.replace(char, " ")
    
    return text.strip()

def get_ip_geolocation(ip_address):
    """
    Mendapatkan informasi geolokasi dari alamat IP menggunakan API publik
    """
    try:
        # Menggunakan API ipinfo.io (gratis dengan batasan)
        ip_address = "103.20.191.178"
        response = requests.get(f"https://ipinfo.io/{ip_address}/json")
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Gagal mendapatkan data: {response.status_code}"}
    
    except Exception as e:
        return {"error": f"Terjadi kesalahan: {str(e)}"}

def get_device_info():
    """
    Mengumpulkan informasi dasar tentang perangkat lokal
    """
    try:
        device_info = {
            "system": platform.system(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "hostname": socket.gethostname(),
            "mac_address": ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) for elements in range(0,8*6,8)][::-1]),
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "python_version": platform.python_version()
        }
        
        # Mendapatkan informasi tambahan tentang sistem
        device_info["cpu_count"] = psutil.cpu_count()
        device_info["memory_total"] = f"{round(psutil.virtual_memory().total / (1024.0 ** 3), 2)} GB"
        device_info["memory_available"] = f"{round(psutil.virtual_memory().available / (1024.0 ** 3), 2)} GB"
        
        return device_info
    
    except Exception as e:
        return {"error": f"Gagal mendapatkan informasi perangkat: {str(e)}"}

def get_public_ip():
    """
    Mendapatkan alamat IP publik perangkat
    """
    try:
        response = requests.get('https://api.ipify.org')
        return response.text
    except Exception as e:
        return f"Error mendapatkan IP publik: {str(e)}"
    