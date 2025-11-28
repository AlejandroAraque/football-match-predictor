import os
import time
import random
import unicodedata
import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO

class LaLigaScraper:
    """
    Scraper profesional para datos de La Liga (FBref).
    Extrae partidos, estadísticas de tiro y posición en tabla,
    realizando el cruce de datos (ETL) al vuelo.
    """
    
    def __init__(self, start_year, end_year, output_dir="data/raw"):
        self.years = list(range(start_year, end_year, -1))
        self.output_dir = output_dir
        self.base_url = "https://fbref.com"
        self.standings_url = "https://fbref.com/en/comps/12/La-Liga-Stats"
        
        # Headers rotatorios o fijos para evitar bloqueos 403
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.google.com/'
        }
        
        # Crear estructura de carpetas automáticamente
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _clean_text(text):
        """Utilidad para normalizar nombres (eliminar acentos)."""
        if not isinstance(text, str): return text
        nfkd_form = unicodedata.normalize('NFKD', text)
        return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

    def _make_request(self, url):
        """Gestor de peticiones con 'Exponential Backoff' y manejo de errores."""
        delay = random.uniform(5, 8) # Pausa aleatoria para parecer humano
        time.sleep(delay)
        
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 429:
                print("⚠️ Demasiadas peticiones. Pausando 60s...")
                time.sleep(60)
                response = requests.get(url, headers=self.headers)
            
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"❌ Error descargando {url}: {e}")
            return None

    def _get_season_standings(self, soup):
        """
        Extrae la tabla de clasificación de la temporada.
        Retorna: DataFrame con stats (W, D, L, xG...) y URLs de los equipos.
        """
        table = soup.select('table.stats_table')[0]
        links = [l.get("href") for l in table.find_all('a')]
        team_urls = [f"{self.base_url}{l}" for l in links if '/squads/' in l]
        # Eliminar duplicados manteniendo el orden
        team_urls = list(dict.fromkeys(team_urls))

        # Procesar filas de la tabla
        rows = table.find_all('tr')[1:] 
        team_stats = []
        
        for row in rows:
            if not row.find('td'): continue # Saltar cabeceras intermedias
            
            # Extraer posición (rank)
            rank_th = row.find('th', {'data-stat': 'rank'})
            position = int(rank_th.get('csk')) if rank_th and rank_th.get('csk') else None
            
            cols = row.find_all('td')
            team_name = self._clean_text(cols[0].text.strip())
            
            # Corrección específica (Hardcoded fix)
            if team_name == "Betis": team_name = "Real Betis"

            # Recopilar métricas clave
            try:
                stats = {
                    'Team': team_name,
                    'Position': position,
                    'W': int(cols[2].text.strip() or 0),
                    'D': int(cols[3].text.strip() or 0),
                    'L': int(cols[4].text.strip() or 0),
                    'GF': int(cols[5].text.strip() or 0),
                    'GA': int(cols[6].text.strip() or 0),
                    'Pts': int(cols[8].text.strip() or 0),
                    'xG': float(cols[10].text.strip() or 0),
                    'xGA': float(cols[11].text.strip() or 0),
                }
                team_stats.append(stats)
            except (ValueError, IndexError):
                continue

        return pd.DataFrame(team_stats), team_urls

    def _process_team_match_data(self, team_url, season_stats):
        """
        Procesa los partidos de un equipo específico:
        1. Descarga partidos (Fixtures)
        2. Descarga disparos (Shooting)
        3. Merge de ambas tablas
        4. Enriquece con datos del rival (Opponent Stats)
        """
        html = self._make_request(team_url)
        if not html: return None

        # 1. Scores & Fixtures
        try:
            matches = pd.read_html(StringIO(html), match="Scores & Fixtures")[0]
        except ValueError: return None

        # 2. Shooting
        soup = BeautifulSoup(html, "html.parser")
        links = [l.get("href") for l in soup.find_all('a') if l.get("href") and 'all_comps/shooting/' in l.get("href")]
        
        if links:
            shooting_url = f"{self.base_url}{links[0]}"
            html_shooting = self._make_request(shooting_url)
            if html_shooting:
                try:
                    shooting = pd.read_html(StringIO(html_shooting), match="Shooting")[0]
                    shooting.columns = shooting.columns.droplevel(0) # Eliminar multi-index
                    
                    # 3. Merge
                    matches = matches.merge(
                        shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], 
                        on="Date", how="left"
                    )
                except ValueError: pass

        # Filtrar solo La Liga
        if 'Comp' in matches.columns:
            matches = matches[matches['Comp'] == 'La Liga'].copy()
        else:
            return None

        # 4. Enriquecimiento de datos (Feature Engineering al vuelo)
        team_name = self._clean_text(team_url.split("/")[-1].replace("-Stats", "").replace("-", " "))
        if team_name == "Betis": team_name = "Real Betis"

        # Añadir stats del propio equipo (Season Stats)
        team_season_info = season_stats[season_stats['Team'] == team_name]
        if not team_season_info.empty:
            info = team_season_info.iloc[0]
            for col in ['Position', 'W', 'D', 'L', 'GF', 'GA', 'xG', 'xGA']:
                matches[f'Team_{col}'] = info[col]
        
        matches['Team'] = team_name

        # Añadir stats del rival (Opponent Stats)
        matches['Opponent'] = matches['Opponent'].apply(self._clean_text)
        matches['Opponent'] = matches['Opponent'].replace("Betis", "Real Betis")

        # Cruzar datos con el DataFrame de standings para rellenar datos del rival
        # Nota: Hacemos un merge left con season_stats usando 'Opponent' como clave
        matches = matches.merge(
            season_stats.add_prefix('Opp_'), 
            left_on='Opponent', 
            right_on='Opp_Team', 
            how='left'
        )
        
        return matches

    def run(self):
        """Ejecuta el ciclo principal de scraping."""
        all_seasons_data = []
        current_url = self.standings_url

        for year in self.years:
            print(f"🔄 Procesando Temporada {year}...")
            
            html = self._make_request(current_url)
            if not html: break
            
            soup = BeautifulSoup(html, "html.parser")
            
            # Paso 1: Obtener tabla de posiciones y stats generales
            standings_df, team_urls = self._get_season_standings(soup)
            print(f"   📊 Tabla de posiciones extraída ({len(standings_df)} equipos).")

            # Paso 2: Iterar equipos
            for team_url in team_urls:
                print(f"   Downloading: {team_url.split('/')[-1]}...")
                match_data = self._process_team_match_data(team_url, standings_df)
                
                if match_data is not None and not match_data.empty:
                    match_data['Season'] = year
                    all_seasons_data.append(match_data)

            # Paso 3: Preparar URL año anterior
            try:
                prev = soup.select("a.prev")[0].get("href")
                current_url = f"{self.base_url}{prev}"
            except IndexError:
                print("🏁 No hay más temporadas previas.")
                break

        # Guardado final unificado
        if all_seasons_data:
            full_df = pd.concat(all_seasons_data, ignore_index=True)
            
            # Limpieza final de columnas basura
            cols_to_drop = ["Match Report", "Notes", "Opp_Team"]
            full_df.drop(columns=[c for c in cols_to_drop if c in full_df.columns], inplace=True)
            
            save_path = os.path.join(self.output_dir, "laliga_full_dataset.csv")
            full_df.to_csv(save_path, index=False)
            print(f"✅ Éxito: Dataset guardado en {save_path}")
        else:
            print("❌ No se generaron datos.")

if __name__ == "__main__":
    # Configuración de ejecución
    scraper = LaLigaScraper(start_year=2024, end_year=2020)
    scraper.run()