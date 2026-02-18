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

    # Jalankan proses robot di latar belakang
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # gabung error ke output biar kelihatan semua
        text=True,
        bufsize=1,                 # baca baris per baris (penting!)
        universal_newlines=True
    )

    # Kosongin kotak teks dulu
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, f"Memproses: {os.path.basename(audio_file)}\n\n")
    output_text.see(tk.END)

    # Baca output robot baris per baris (live!)
    def read_output():
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            output_text.insert(tk.END, line)
            output_text.see(tk.END)  # scroll otomatis ke bawah

        # Setelah selesai
        return_code = process.returncode
        if return_code == 0:
            messagebox.showinfo("Selesai!", f"Rapat sudah ditulis!\nCek folder: data/output/{os.path.basename(audio_file).split('.')[0]}")
        else:
            messagebox.showerror("Waduh Error", f"Ada masalah nih (kode {return_code})")
        
        start_button.config(state="normal")
        status_label.config(text="Siap untuk rapat berikutnya!")

    # Jalankan baca output di thread terpisah biar GUI nggak beku
    import threading
    thread = threading.Thread(target=read_output)
    thread.start()

stop_requested = False  # seperti bendera merah kecil

def stop_processing():
    global stop_requested
    stop_requested = True
    status_label.config(text="Sedang berhenti... tunggu sebentar ya!")
    stop_button.config(state="disabled")

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

# Tombol STOP (awalnya disembunyikan)
stop_button = tk.Button(root, text="STOP SEKARANG!", font=("Arial", 14, "bold"), 
                        command=stop_processing, bg="#F44336", fg="white", state="disabled")
stop_button.pack(pady=5)

# Status
status_label = tk.Label(root, text="Ready!", font=("Arial", 10), bg="#1e1e1e")
status_label.pack(pady=5)

# Kotak lihat progres (seperti layar kecil robot)
output_text = scrolledtext.ScrolledText(root, height=15, width=70, font=("Consolas", 10))
output_text.pack(pady=10)

root.mainloop()