import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import subprocess
import os
import threading

# Ini alamat rumah proyekmu (ubah kalau beda)
PROJECT_DIR = "/Users/670141/Documents/Python/Speech To Text"

def run_pipeline(audio_file):
    config_path = os.path.join(PROJECT_DIR, "configs", "config.yaml")
    cmd = [
        "python3",
        os.path.join(PROJECT_DIR, "scripts", "process_audio.py"),
        "--config", config_path
    ]

    # Kita nggak pakai --resume biar mulai dari nol setiap kali klik
    # Kalau mau resume, tambahin "--resume" di list di atas

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Baca output sambil jalan (biar kelihatan progres)
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, f"Memproses: {os.path.basename(audio_file)}\n\n")
    
    for line in process.stdout:
        output_text.insert(tk.END, line)
        output_text.see(tk.END)
    
    process.wait()
    
    if process.returncode == 0:
        messagebox.showinfo("Done", f"Audio transcribed!\nCek folder: data/output/{os.path.basename(audio_file).split('.')[0]}")
    else:
        error = process.stderr.read()
        messagebox.showerror("Error", f"Description:\n{error}")

def start_processing():
    start_button.config(state="disabled")
    status_label.config(text="Transcribe in progress, wait until it finished!")
    
    # Jalankan di thread terpisah biar tombol nggak beku
    thread = threading.Thread(target=run_pipeline, args=(selected_file.get(),))
    thread.start()
    
    # Cek selesai thread
    check_thread(thread)

def check_thread(thread):
    if thread.is_alive():
        root.after(1000, check_thread, thread)
    else:
        start_button.config(state="normal")
        status_label.config(text="Ready for next audio")

def choose_file():
    file_path = filedialog.askopenfilename(
        title="Choose Audio",
        filetypes=[("Audio files", "*.m4a *.mp3 *.wav *.aac *.ogg")]
    )
    if file_path:
        selected_file.set(file_path)
        file_label.config(text=f"Choosen Audio: {os.path.basename(file_path)}")
        start_button.config(state="normal")

# Bikin jendela utama (kotak mainan)
root = tk.Tk()
root.title("Meeting Transciber v2.0")
root.geometry("600x500")
root.configure(bg="#1e1e1e")

# Label judul
tk.Label(root, text="Meeting Transcriber Offline v2.0", font=("Arial", 16, "bold"), bg="#1e1e1e").pack(pady=10)

# Tombol pilih file
tk.Button(root, text="Choose Audio", font=("Arial", 12), command=choose_file, bg="#4CAF50", fg="black").pack(pady=10)

# Tampil file yang dipilih
selected_file = tk.StringVar()
file_label = tk.Label(root, text="File not choosen yet", font=("Arial", 10), bg="#1e1e1e")
file_label.pack(pady=5)

# Tombol mulai
start_button = tk.Button(root, text="Start Trascribing!", font=("Arial", 14, "bold"), 
                         command=start_processing, bg="#2196F3", fg="white", state="disabled")
start_button.pack(pady=20)

# Status
status_label = tk.Label(root, text="Ready!", font=("Arial", 10), bg="#1e1e1e")
status_label.pack(pady=5)

# Kotak lihat progres (seperti layar kecil robot)
output_text = scrolledtext.ScrolledText(root, height=15, width=70, font=("Consolas", 10))
output_text.pack(pady=10)

root.mainloop()