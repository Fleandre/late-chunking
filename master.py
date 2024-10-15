import json
import time
import itertools
from tasks import compute_task
from chunked_pooling.chunked_eval_tasks import *
from eval_utils import *


def main():
    tasks, benchmark = generate_tasks()
    result_tasks = []

    for task_str in tasks:
        result = compute_task.delay(task_str)
        result_tasks.append({"task_id": task_str, "result": result})

    while result_tasks:
        for result_task in result_tasks:
            is_ready = False
            try:
                is_ready = result_task["result"].ready()
            except Exception as e:
                continue
            if not is_ready:
                continue
            # 任务已完成
            result = result_task["result"].get()
            if result is None:
                continue

            benchmark.append(result)

            print(f"Task: {result_task['task_id']} finished.")

            # 移除已完成任务
            result_tasks.remove(result_task)

        # 集中写入一次结果
        with open(f"{OUTPUT_DIR}/benchmark.json", "w", encoding="utf-8") as f:
            json.dump(benchmark, f, ensure_ascii=False, indent=4)

        time.sleep(10)  # 轮询间隔时间


if __name__ == "__main__":
    main()
