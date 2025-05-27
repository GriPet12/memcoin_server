import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Memecoin Graduation Predictor",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Memecoin Graduation Predictor")
st.markdown("Прогнозування ймовірності graduation мемкоїна на основі транзакційних даних")


# Кешування моделі
@st.cache_resource
def load_model():
    try:
        # Завантажте вашу натреновану модель
        model = joblib.load('memecoin_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Модель не знайдена. Спочатку збережіть модель як 'memcoin_model.pkl'")
        return None
    except Exception as e:
        st.error(f"Помилка завантаження моделі: {str(e)}")
        return None


def preprocess_input(data):
    """Попередня обробка вхідних даних"""
    # Очищення назв колонок
    data.columns = data.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)
    data.columns = data.columns.str.replace('__+', '_', regex=True)
    data.columns = data.columns.str.strip('_')

    # Заповнення пропусків
    data = data.fillna(0)

    return data


# Sidebar для введення даних
st.sidebar.header("📊 Введіть дані токену")

# Основні характеристики
tx_count = st.sidebar.number_input("Кількість транзакцій", min_value=0, value=100)
unique_wallets = st.sidebar.number_input("Унікальні гаманці", min_value=0, value=50)
quote_coin_sum = st.sidebar.number_input("Загальна сума quote coin", min_value=0.0, value=1000.0)
quote_coin_mean = st.sidebar.number_input("Середня сума quote coin", min_value=0.0, value=20.0)
base_coin_sum = st.sidebar.number_input("Загальна сума base coin", min_value=0.0, value=50000.0)
base_coin_mean = st.sidebar.number_input("Середня сума base coin", min_value=0.0, value=1000.0)
buy_count = st.sidebar.number_input("Кількість покупок", min_value=0, value=70)
sell_count = st.sidebar.number_input("Кількість продажів", min_value=0, value=30)
decimals = st.sidebar.number_input("Decimals", min_value=0, value=6)
bundle_size = st.sidebar.number_input("Bundle size", min_value=0.0, value=1.0)

# Обчислення додаткових характеристик
buy_sell_ratio = buy_count / (sell_count + 1e-6) if sell_count > 0 else buy_count

# Створення датафрейма з введеними даними
input_data = pd.DataFrame({
    'tx_idx_count': [tx_count],
    'signing_wallet_nunique': [unique_wallets],
    'quote_coin_amount_sum': [quote_coin_sum],
    'quote_coin_amount_mean': [quote_coin_mean],
    'base_coin_amount_sum': [base_coin_sum],
    'base_coin_amount_mean': [base_coin_mean],
    'buy_count': [buy_count],
    'sell_count': [sell_count],
    'buy_sell_ratio': [buy_sell_ratio],
    'decimals': [decimals],
    'bundle_size': [bundle_size]
})

# Завантаження моделі
model, metadata = load_model()

# Показати інформацію про модель
if model is not None and metadata is not None:
    st.sidebar.success("✅ Модель завантажена успішно")
    with st.sidebar.expander("ℹ️ Інформація про модель"):
        st.write(f"**Тип моделі:** {metadata['model_type']}")
        st.write(f"**Кількість фічей:** {metadata['n_features']}")
        st.write(f"**CV LogLoss:** {metadata['cv_score']:.4f}")
        st.write(f"**Фічі:** {', '.join(metadata['feature_names'][:5])}...")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Поточні дані токену")
    st.dataframe(input_data.round(4))

with col2:
    if model is not None and st.button("🔮 Зробити прогноз", type="primary"):
        try:
            # Обробка даних
            processed_data = preprocess_input(input_data.copy())

            # Прогноз
            probability = model.predict_proba(processed_data)[0][1]

            st.subheader("🎯 Результат прогнозу")

            # Відображення результату з кольоровим індикатором
            if probability > 0.7:
                st.success(f"Висока ймовірність graduation: {probability:.1%}")
                st.balloons()
            elif probability > 0.4:
                st.warning(f"Середня ймовірність graduation: {probability:.1%}")
            else:
                st.error(f"Низька ймовірність graduation: {probability:.1%}")

            # Прогрес-бар
            st.progress(probability)

            # Додаткова інформація
            st.info(f"""
            **Інтерпретація:**
            - Ймовірність > 70%: Токен має високі шанси на graduation
            - Ймовірність 40-70%: Помірні шанси, потрібен додатковий аналіз
            - Ймовірність < 40%: Низькі шанси на graduation
            """)

        except Exception as e:
            st.error(f"Помилка при прогнозуванні: {str(e)}")

# Завантаження CSV файлу
st.subheader("📁 Пакетний прогноз з CSV")
uploaded_file = st.file_uploader("Завантажте CSV файл з даними токенів", type="csv")

if uploaded_file is not None and model is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Завантажені дані:")
        st.dataframe(df.head())

        if st.button("Зробити пакетний прогноз"):
            processed_df = preprocess_input(df.copy())
            predictions = model.predict_proba(processed_df)[:, 1]

            df['graduation_probability'] = predictions
            df['prediction'] = (predictions > 0.5).astype(int)

            st.success("Прогноз завершено!")
            st.dataframe(df[['graduation_probability', 'prediction']])

            # Можливість завантажити результати
            csv = df.to_csv(index=False)
            st.download_button(
                label="Завантажити результати CSV",
                data=csv,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Помилка при обробці файлу: {str(e)}")

# Інструкції для користувача
with st.expander("ℹ️ Інструкції по використанню"):
    st.markdown("""
    ### Як використовувати:

    1. **Індивідуальний прогноз**: Введіть параметри токену в sidebar і натисніть "Зробити прогноз"

    2. **Пакетний прогноз**: Завантажте CSV файл з колонками:
       - tx_idx_count
       - signing_wallet_nunique  
       - quote_coin_amount_sum
       - quote_coin_amount_mean
       - base_coin_amount_sum
       - base_coin_amount_mean
       - buy_count
       - sell_count
       - decimals
       - bundle_size

    3. **Інтерпретація результатів**:
       - Значення від 0 до 1 (ймовірність graduation)
       - Чим вище значення, тим більша ймовірність успіху
    """)

st.markdown("---")
st.markdown("Створено для аналізу мемкоїнів 🚀")