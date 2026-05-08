# =============================================================
# criar_banco.py
# Passo 3: Cria o banco clinica.db e importa os dados do CSV
# Execução: python criar_banco.py
# Depende de: pacientes.csv (rode gerar_dados.py antes)
# =============================================================

import sqlite3
import hashlib
import pandas as pd

DB_PATH  = "clinica.db"
CSV_PATH = "pacientes.csv"


def hash_senha(senha: str) -> str:
    """Retorna o hash SHA-256 da senha."""
    return hashlib.sha256(senha.encode()).hexdigest()


def criar_tabelas(cursor: sqlite3.Cursor) -> None:
    """Cria todas as tabelas do sistema."""

    # Tabela de usuários (login)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            user     TEXT    UNIQUE NOT NULL,
            password TEXT    NOT NULL,
            perfil   TEXT    DEFAULT 'medico'
        )
    """)

    # Tabela de pacientes
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pacientes (
            id    INTEGER PRIMARY KEY AUTOINCREMENT,
            nome  TEXT    NOT NULL,
            idade INTEGER NOT NULL
        )
    """)

    # Tabela de exames (histórico de predições)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS exames (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            id_paciente   INTEGER NOT NULL,
            glicose       REAL,
            pressao       REAL,
            imc           REAL,
            colesterol    REAL,
            resultado_ia  TEXT,
            probabilidade REAL,
            data_exame    TEXT DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (id_paciente) REFERENCES pacientes(id)
        )
    """)

    # Tabela de limites de alerta (configurável pelo médico)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS limites_exames (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            exame        TEXT  UNIQUE NOT NULL,
            valor_alerta REAL  NOT NULL,
            unidade      TEXT
        )
    """)


def popular_dados_iniciais(cursor: sqlite3.Cursor) -> None:
    """Insere usuários e limites padrão (ignora se já existirem)."""

    usuarios = [
        ("admin",  hash_senha("admin123"),  "admin"),
        ("medico", hash_senha("medico123"), "medico"),
    ]
    cursor.executemany(
        "INSERT OR IGNORE INTO usuarios (user, password, perfil) VALUES (?,?,?)",
        usuarios,
    )

    limites = [
        ("glicose",    125.0, "mg/dL"),
        ("pressao",     90.0, "mmHg"),
        ("imc",         30.0, "kg/m²"),
        ("colesterol", 200.0, "mg/dL"),
    ]
    cursor.executemany(
        "INSERT OR IGNORE INTO limites_exames (exame, valor_alerta, unidade) VALUES (?,?,?)",
        limites,
    )


def importar_csv(cursor: sqlite3.Cursor, csv_path: str) -> tuple[int, int]:
    """Importa pacientes e exames do CSV para o banco."""
    df = pd.read_csv(csv_path)
    pacientes_count = 0
    exames_count    = 0

    for _, row in df.iterrows():
        cursor.execute(
            "INSERT INTO pacientes (nome, idade) VALUES (?, ?)",
            (row["nome"], int(row["idade"])),
        )
        id_paciente = cursor.lastrowid
        pacientes_count += 1

        resultado = "Alerta" if row["risco"] == 1 else "Normal"
        cursor.execute(
            """INSERT INTO exames
               (id_paciente, glicose, pressao, imc, colesterol, resultado_ia)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (id_paciente, row["glicose"], row["pressao"],
             row["imc"], row["colesterol"], resultado),
        )
        exames_count += 1

    return pacientes_count, exames_count


# --- Main ---
if __name__ == "__main__":
    print(f"🗄️  Criando banco de dados: {DB_PATH}")

    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    criar_tabelas(cursor)
    print("   ✅ Tabelas criadas: usuarios, pacientes, exames, limites_exames")

    popular_dados_iniciais(cursor)
    print("   ✅ Usuários padrão inseridos")
    print("      admin  / admin123")
    print("      medico / medico123")

    pac, ex = importar_csv(cursor, CSV_PATH)
    conn.commit()
    conn.close()

    print(f"   ✅ CSV importado: {pac} pacientes | {ex} exames")
    print(f"\n✅ Banco {DB_PATH} pronto!")
    print(f"\n▶  Próximo passo: streamlit run app.py")
