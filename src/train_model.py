import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report

class MatchPredictor:
    def __init__(self, data_path="data/raw/laliga_full_dataset.csv", models_dir="models"):
        self.data_path = data_path
        self.models_dir = models_dir
        self.predictors = [] 
        
        # Configuración de modelos
        self.models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1),
            "KNN": KNeighborsClassifier(n_neighbors=5)
        }
        
        os.makedirs(self.models_dir, exist_ok=True)

    def load_and_clean_data(self):
        """Carga y limpieza inicial (ETL)."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"❌ Falta el archivo: {self.data_path}")
            
        df = pd.read_csv(self.data_path)
        
        # 1. Normalizar columnas a minúsculas (Date -> date, Opponent -> opponent)
        df.columns = df.columns.str.lower()
        
        # 2. Verificar columna Date
        if "date" not in df.columns:
            raise KeyError(f"No se encuentra la columna 'date'. Columnas disponibles: {list(df.columns)}")
        df["date"] = pd.to_datetime(df["date"])
        
        # 3. Crear Target (Variable a predecir)
        # Si 'result' no existe, lo calculamos con GF y GA
        if "result" in df.columns:
            df["target"] = (df["result"] == "W").astype("int")
        elif "gf" in df.columns and "ga" in df.columns:
             # Si gana (GF > GA) -> 1, si no -> 0
            df["target"] = (df["gf"] > df["ga"]).astype("int")
        else:
            raise KeyError("Faltan columnas para determinar el ganador ('result' o 'gf'/'ga')")

        # 4. Encoding de categóricas
        # Convertimos texto a números. Como tu CSV tiene 'opponent', creamos 'opponent_code'
        df["venue_code"] = df["venue"].astype("category").cat.codes
        df["opponent_code"] = df["opponent"].astype("category").cat.codes
        df["team_code"] = df["team"].astype("category").cat.codes
        
        # 5. Features temporales
        # Si no hay hora, ponemos las 20:00 por defecto
        if "time" in df.columns:
            df["hour"] = df["time"].astype(str).str.replace(":.+", "", regex=True)
            df["hour"] = pd.to_numeric(df["hour"], errors='coerce').fillna(20).astype(int)
        else:
            df["hour"] = 20 # Valor por defecto

        df["day_code"] = df["date"].dt.dayofweek
        
        return df

    def add_rolling_averages(self, df):
        """Genera medias móviles de los últimos 3 partidos."""
        # Estas son las columnas numéricas de tu CSV que nos interesan
        potential_cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
        
        # Filtrar solo las que existen
        cols_available = [c for c in potential_cols if c in df.columns]
        
        if not cols_available:
            print("⚠️ No hay estadísticas suficientes para medias móviles.")
            self.predictors = ["venue_code", "opponent_code", "hour", "day_code", "team_code"]
            return df

        new_cols = [f"{c}_rolling" for c in cols_available]
        
        # Función de rolling (agrupando por equipo)
        # Usamos group_keys=False para evitar índices duplicados
        df_rolling = df.groupby("team", group_keys=False)[cols_available].apply(
            lambda x: x.rolling(3, closed='left').mean()
        )
        
        df_rolling.columns = new_cols
        
        # Concatenar al dataframe original
        df = pd.concat([df, df_rolling], axis=1)
        
        # Eliminar filas con NaN (los primeros 3 partidos de cada equipo)
        df = df.dropna()
        
        # Definir predictores finales (AQUÍ ESTABA EL ERROR: usamos opponent_code)
        self.predictors = ["venue_code", "opponent_code", "hour", "day_code", "team_code"] + new_cols
        return df

    def train_and_compare(self):
        print("🔄 Cargando y procesando datos...")
        try:
            df = self.load_and_clean_data()
            df = self.add_rolling_averages(df)
        except Exception as e:
            print(f"❌ Error preparando datos: {e}")
            return

        # Split temporal (Último año = Test)
        cutoff_date = df["date"].max() - pd.Timedelta(days=365)
        train = df[df["date"] < cutoff_date]
        test = df[df["date"] >= cutoff_date]
        
        print(f"📊 Datos listos. Train: {len(train)} | Test: {len(test)}")
        
        if len(train) == 0:
            print("❌ Error: Set de entrenamiento vacío.")
            return

        results = []

        for name, model in self.models.items():
            print(f"🧠 Entrenando {name}...")
            # Entrenamos usando self.predictors
            model.fit(train[self.predictors], train["target"])
            
            preds = model.predict(test[self.predictors])
            
            acc = accuracy_score(test["target"], preds)
            prec = precision_score(test["target"], preds, zero_division=0)
            
            results.append({"Modelo": name, "Accuracy": acc, "Precision": prec})
            
            # Guardar modelo
            filename = os.path.join(self.models_dir, f"{name.lower()}_model.pkl")
            joblib.dump(model, filename)
        
        results_df = pd.DataFrame(results).sort_values(by="Precision", ascending=False)
        
        print("\n" + "="*60)
        print("🏆 TABLA DE RESULTADOS")
        print("="*60)
        print(results_df.to_string(index=False))
        print("="*60)
        
        if not results_df.empty:
            best = results_df.iloc[0]
            print(f"✅ Mejor modelo: {best['Modelo']} (Precisión: {best['Precision']:.2%})")

if __name__ == "__main__":
    predictor = MatchPredictor()
    predictor.train_and_compare()