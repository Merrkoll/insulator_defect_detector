import os
import sys
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

class LabelerApp:
    def __init__(self, src_dir, out_dir):
        self.src_dir = Path(src_dir)
        self.out_dir = Path(out_dir)
        self.good_dir = self.out_dir / 'good'
        self.defect_dir = self.out_dir / 'defect'
        self.skip_dir = self.out_dir / 'skip'
        self.good_dir.mkdir(parents=True, exist_ok=True)
        self.defect_dir.mkdir(parents=True, exist_ok=True)
        self.skip_dir.mkdir(parents=True, exist_ok=True)

        self.images = [p for p in sorted(self.src_dir.iterdir()) if p.is_file() and p.suffix.lower() in VALID_EXTS]
        self.index = 0
        self.history = []
        self.tk_img = None

        self.root = tk.Tk()
        self.root.title('Quick Labeler: 1=good, 2=defect, 0=skip, Backspace=undo, Q=quit')
        self.root.geometry('1200x850')
        self.root.configure(bg='black')

        self.info = tk.Label(self.root, text='', fg='white', bg='black', font=('Arial', 14))
        self.info.pack(pady=8)

        self.img_label = tk.Label(self.root, bg='black')
        self.img_label.pack(expand=True)

        self.help_label = tk.Label(
            self.root,
            text='Клавиши: 1 — исправный (good), 2 — дефектный (defect), 0/Space — пропустить, Backspace — отменить последнее действие, Q/Esc — выйти',
            fg='white', bg='black', font=('Arial', 12)
        )
        self.help_label.pack(pady=8)

        self.root.bind('1', lambda e: self.move_current('good'))
        self.root.bind('2', lambda e: self.move_current('defect'))
        self.root.bind('0', lambda e: self.move_current('skip'))
        self.root.bind('<space>', lambda e: self.move_current('skip'))
        self.root.bind('<BackSpace>', lambda e: self.undo())
        self.root.bind('q', lambda e: self.quit())
        self.root.bind('<Escape>', lambda e: self.quit())

        if not self.images:
            messagebox.showinfo('Нет изображений', f'В папке {self.src_dir} не найдено изображений.')
            self.root.destroy()
            return

        self.show_current()
        self.root.mainloop()

    def fit_image(self, img, max_w=1100, max_h=700):
        w, h = img.size
        scale = min(max_w / w, max_h / h, 1.0)
        new_size = (int(w * scale), int(h * scale))
        return img.resize(new_size, Image.LANCZOS)

    def show_current(self):
        if self.index >= len(self.images):
            self.info.config(text='Разметка завершена. Все изображения обработаны.')
            self.img_label.config(image='')
            return

        img_path = self.images[self.index]
        processed = self.index
        total = len(self.images)
        self.info.config(text=f'{processed + 1}/{total} | {img_path.name}')

        try:
            img = Image.open(img_path).convert('RGB')
            img = self.fit_image(img)
            self.tk_img = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.tk_img)
        except Exception as e:
            self.info.config(text=f'Ошибка открытия {img_path.name}: {e}')
            self.index += 1
            self.show_current()

    def unique_dest(self, folder, name):
        dest = folder / name
        if not dest.exists():
            return dest
        stem = Path(name).stem
        suffix = Path(name).suffix
        i = 1
        while True:
            candidate = folder / f'{stem}_{i}{suffix}'
            if not candidate.exists():
                return candidate
            i += 1

    def move_current(self, target):
        if self.index >= len(self.images):
            return
        src = self.images[self.index]
        folder = {'good': self.good_dir, 'defect': self.defect_dir, 'skip': self.skip_dir}[target]
        dest = self.unique_dest(folder, src.name)
        shutil.move(str(src), str(dest))
        self.history.append((str(dest), str(src), self.index))
        self.index += 1
        self.show_current()

    def undo(self):
        if not self.history:
            return
        moved_from, moved_to, old_index = self.history.pop()
        shutil.move(moved_from, moved_to)
        self.index = old_index
        self.show_current()

    def quit(self):
        self.root.destroy()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Использование: python quick_labeler.py <src_dir> <out_dir>')
        print('Пример: python quick_labeler.py data_raw data')
        sys.exit(1)

    src_dir = sys.argv[1]
    out_dir = sys.argv[2]
    LabelerApp(src_dir, out_dir)