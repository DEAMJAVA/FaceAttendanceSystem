import csv
import os
from datetime import datetime

import config


class AttendanceLog:
    def __init__(self):
        self._marked_today_cache = None
        self._cache_date = None

    def _file_for_today(self):
        date_str = datetime.now().strftime("%d-%m-%Y")
        day_dir = os.path.join(config.ATTENDANCE_DIR, date_str)
        os.makedirs(day_dir, exist_ok=True)
        file_path = os.path.join(day_dir, "attendance.csv")
        if not os.path.exists(file_path):
            with open(file_path, "w", newline="") as f:
                csv.writer(f).writerow(["Name", "Date", "Time"])
        return file_path, date_str

    def _already_marked(self, file_path, date_str, name):
        if self._cache_date != date_str:
            self._marked_today_cache = set()
            with open(file_path, "r", newline="") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    if len(row) >= 2:
                        self._marked_today_cache.add(row[0])
            self._cache_date = date_str
        return name in self._marked_today_cache

    def mark(self, name):
        file_path, date_str = self._file_for_today()
        if self._already_marked(file_path, date_str, name):
            return False

        time_str = datetime.now().strftime("%I:%M:%S %p")
        with open(file_path, "a", newline="") as f:
            csv.writer(f).writerow([name, date_str, time_str])
        self._marked_today_cache.add(name)
        print(f"[attendance] Marked {name} at {time_str}")
        return True