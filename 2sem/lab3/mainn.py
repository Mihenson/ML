from neo4j import GraphDatabase
import numpy as np

# Подключение к базе данных Neo4j
URI = "bolt://localhost:7687"
USER = "admin"
PASSWORD = "admin12345"
STEPS = 5  # Количество шагов симуляции
np.random.seed(42)

class WarehouseRobotControlSystem:
    def __init__(self, uri, user, password):
        # Инициализация подключения к базе данных Neo4j
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        # Закрытие подключения к базе данных
        self.driver.close()

    def get_action(self, storage_condition, order_size, robot_status):
        # Запрос в базу данных для получения действия робота и склада
        query = (
            "MATCH (st:StorageCondition {value: $storage_condition})-[a:ACTION]->"
            "(o:OrderCondition {value: $order_size})-[b:ACTION]->(r:RobotCondition {value: $robot_status}) "
            "RETURN a.name AS storage_action, b.name AS robot_action"
        )

        with self.driver.session() as session:
            result = session.run(
                query,
                storage_condition=storage_condition,
                order_size=order_size,
                robot_status=robot_status
            )

            records = result.data()
            if records:
                # Выбираем первое совпадение
                record = records[0]
                return record["storage_action"], record["robot_action"]
            # Возвращаем дефолтные действия, если не найдены соответствующие в базе
            return "No action", "No action"

    def fuzzify_storage_condition(self, stock_level):
        """Фуззификация уровня запасов на складе (0-100)"""
        if stock_level <= 33:
            return "LowStock"
        elif stock_level <= 66:
            return "MediumStock"
        else:
            return "HighStock"

    def fuzzify_order_size(self, order_size):
        """Фуззификация размера заказа (0-100)"""
        if order_size <= 33:
            return "SmallOrder"
        elif order_size <= 66:
            return "MediumOrder"
        else:
            return "LargeOrder"

    def fuzzify_robot_status(self, battery_level):
        """Фуззификация статуса робота (0-100)"""
        if battery_level <= 33:
            return "Charging"
        elif battery_level <= 66:
            return "Idle"
        else:
            return "Busy"

    def simulate(self, initial_storage_condition, initial_order_size, initial_battery_level, steps=5):
        # Симуляция работы склада и робота на протяжении заданного количества шагов
        storage_condition = self.fuzzify_storage_condition(initial_storage_condition)
        order_size = self.fuzzify_order_size(initial_order_size)
        robot_status = self.fuzzify_robot_status(initial_battery_level)
        battery_level = initial_battery_level

        for step in range(steps):
            # Получаем действия для склада и робота
            storage_action, robot_action = self.get_action(storage_condition, order_size, robot_status)

            print(
                f"Шаг {step+1}: StorageCondition={storage_condition}, OrderSize={order_size}, "
                f"RobotStatus={robot_status}, BatteryLevel={battery_level}, StorageAction={storage_action}, RobotAction={robot_action}"
            )

            # Обновляем статус робота
            if robot_status == "Charging":
                battery_level += np.random.uniform(7.0, 10.0)
            elif robot_status == "Idle":
                battery_level -= np.random.uniform(1.5, 3.0)
            elif robot_status == "Busy":
                battery_level -= np.random.uniform(3.0, 7.0)

            # Ограничиваем заряд батареи между 0 и 100
            battery_level = max(0, min(battery_level, 100))

            # Обновляем состояние склада после изменений
            storage_condition = self.fuzzify_storage_condition(np.random.randint(0, 101))

            # Обновляем размер заказа
            order_size = self.fuzzify_order_size(np.random.randint(0, 101))


# Инициализация системы управления
if __name__ == "__main__":
    system = WarehouseRobotControlSystem(URI, USER, PASSWORD)

    try:
        print("\nСценарий 1: Низкий запас, маленький заказ, робот занят")
        system.simulate(initial_storage_condition=20, initial_order_size=30, initial_battery_level=50, steps=STEPS)

        print("\nСценарий 2: Средний запас, крупный заказ, робот свободен")
        system.simulate(initial_storage_condition=50, initial_order_size=90, initial_battery_level=80, steps=STEPS)

        print("\nСценарий 3: Высокий запас, маленький заказ, робот заряжается")
        system.simulate(initial_storage_condition=80, initial_order_size=30, initial_battery_level=25, steps=STEPS)

    finally:
        system.close()