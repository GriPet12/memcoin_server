"""
Простий тест імпортів та моделі
"""


def test_imports():
    """Тестування базових імпортів"""
    try:
        print("🔍 Тестування імпортів...")

        import numpy as np
        print(f"✅ NumPy {np.__version__}")

        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")

        import lightgbm as lgb
        print(f"✅ LightGBM {lgb.__version__}")

        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")

        import joblib
        print(f"✅ Joblib {joblib.__version__}")

        return True

    except Exception as e:
        print(f"❌ Помилка імпорту: {str(e)}")
        return False


def test_model_loading():
    """Тестування завантаження моделі"""
    try:
        print("\n📥 Тестування завантаження моделі...")

        import joblib
        import os

        if not os.path.exists('memecoin_model.pkl'):
            print("❌ Файл моделі не знайдено!")
            return False

        model = joblib.load('memecoin_model.pkl')
        print("✅ Модель завантажена успішно!")

        # Простий тест
        import numpy as np
        import pandas as pd

        # Мінімальні тестові дані
        test_data = pd.DataFrame({
            'Unnamed__0': [1.0],
            'is_valid': [1.0],
            'decimals': [6.0],
            'bundle_size': [1.0],
            'tx_idx_count': [100.0],
            'signing_wallet_nunique': [50.0],
            'quote_coin_amount_sum': [1000.0],
            'quote_coin_amount_mean': [20.0],
            'base_coin_amount_sum': [50000.0],
            'base_coin_amount_mean': [1000.0],
            'buy_count': [70.0],
            'sell_count': [30.0],
            'buy_sell_ratio': [2.33]
        })

        # Прогноз
        try:
            prediction = model.predict_proba(test_data)
            print(f"🎯 Тестовий прогноз: {prediction[0][1]:.4f}")
            print("✅ Модель працює!")
            return True
        except Exception as pred_error:
            print(f"❌ Помилка прогнозу: {str(pred_error)}")
            print("🔍 Перевірте відповідність фічей...")
            return False

    except Exception as e:
        print(f"❌ Помилка завантаження: {str(e)}")
        return False


if __name__ == "__main__":
    print("🚀 Простий тест системи")
    print("=" * 40)

    # Тест імпортів
    imports_ok = test_imports()

    if imports_ok:
        # Тест моделі
        model_ok = test_model_loading()

        if model_ok:
            print("\n🎉 Все працює! Можете запускати Streamlit")
        else:
            print("\n🔧 Проблема з моделлю")
    else:
        print("\n🔧 Проблема з залежностями")