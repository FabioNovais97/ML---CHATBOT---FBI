# =============================================================
# app.py
# Interface principal do sistema de triagem clínica
# Execução: streamlit run app.py
# Depende de: clinica.db e modelo_risco.pkl
# =============================================================

import sqlite3
import hashlib

import joblib
import pandas as pd
import streamlit as st

# ── Configuração da Página ────────────────────────────────────
st.set_page_config(
    page_title="Clínica IA — Triagem de Risco",
    page_icon="🏥",
    layout="wide",
)

DB_PATH    = "clinica.db"
MODEL_PATH = "modelo_risco.pkl"

# Deve ser IDÊNTICO ao FEATURES de treinar_modelo.py
FEATURES = ["glicose", "pressao", "imc", "colesterol"]


# ── Helpers ───────────────────────────────────────────────────

def hash_senha(senha: str) -> str:
    return hashlib.sha256(senha.encode()).hexdigest()


def get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)


@st.cache_resource
def load_model():
    """Carrega o pipeline .pkl uma única vez (fica em cache)."""
    return joblib.load(MODEL_PATH)


# ── Tela: Login ───────────────────────────────────────────────

def tela_login() -> None:
    st.markdown(
        "<h1 style='text-align:center;'>🏥 Sistema de Triagem Clínica</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:gray;'>Powered by Machine Learning</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.subheader("🔐 Login")
        usuario = st.text_input("Usuário", placeholder="ex: medico")
        senha   = st.text_input("Senha", type="password")
        entrar  = st.button("Entrar", use_container_width=True, type="primary")

        if entrar:
            conn   = get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, user, perfil FROM usuarios WHERE user=? AND password=?",
                (usuario, hash_senha(senha)),
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                st.session_state["logado"]  = True
                st.session_state["usuario"] = row[1]
                st.session_state["perfil"]  = row[2]
                st.rerun()
            else:
                st.error("❌ Usuário ou senha incorretos.")

        st.caption("Credenciais de teste:  medico / medico123")


# ── Tela: Dashboard ───────────────────────────────────────────

def tela_dashboard() -> None:
    st.title("📊 Painel Geral")

    conn      = get_conn()
    total_pac = pd.read_sql("SELECT COUNT(*) AS n FROM pacientes", conn).iloc[0, 0]
    total_ex  = pd.read_sql("SELECT COUNT(*) AS n FROM exames",    conn).iloc[0, 0]
    alertas   = pd.read_sql(
        "SELECT COUNT(*) AS n FROM exames WHERE resultado_ia='Alerta'", conn
    ).iloc[0, 0]
    conn.close()

    c1, c2, c3 = st.columns(3)
    c1.metric("👥 Pacientes Cadastrados", total_pac)
    c2.metric("🧪 Exames Realizados",     total_ex)
    c3.metric(
        "⚠️ Em Alerta",
        alertas,
        delta=f"{round(alertas / total_ex * 100, 1)}% do total" if total_ex else "0%",
        delta_color="inverse",
    )

    st.divider()
    st.subheader("📋 Últimos 10 Exames")

    conn = get_conn()
    df = pd.read_sql(
        """
        SELECT p.nome, e.glicose, e.pressao, e.imc, e.colesterol,
               e.resultado_ia, e.data_exame
        FROM exames e
        JOIN pacientes p ON p.id = e.id_paciente
        ORDER BY e.id DESC
        LIMIT 10
        """,
        conn,
    )
    conn.close()

    def colorir_resultado(val: str) -> str:
        if val == "Alerta":
            return "background-color:#ffcccc; color:#cc0000; font-weight:bold"
        return "background-color:#ccffcc; color:#006600; font-weight:bold"

    st.dataframe(
        df.style.map(colorir_resultado, subset=["resultado_ia"]),
        use_container_width=True,
    )


# ── Tela: Cadastro de Pacientes ───────────────────────────────

def tela_cadastro_pacientes() -> None:
    st.title("👤 Cadastro de Pacientes")

    with st.form("form_paciente", clear_on_submit=True):
        st.subheader("Novo Paciente")
        nome   = st.text_input("Nome completo")
        idade  = st.number_input("Idade", min_value=0, max_value=120, step=1)
        salvar = st.form_submit_button("💾 Cadastrar", type="primary")

        if salvar:
            if not nome.strip():
                st.error("Nome é obrigatório.")
            else:
                conn   = get_conn()
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO pacientes (nome, idade) VALUES (?, ?)",
                    (nome.strip(), int(idade)),
                )
                conn.commit()
                conn.close()
                st.success(f"✅ Paciente '{nome}' cadastrado!")

    st.divider()
    st.subheader("📋 Pacientes Cadastrados")

    conn = get_conn()
    df   = pd.read_sql("SELECT id, nome, idade FROM pacientes ORDER BY id DESC", conn)
    conn.close()
    st.dataframe(df, use_container_width=True)


# ── Tela: Gestão de Limites de Alerta ────────────────────────

def tela_gestao_exames() -> None:
    st.title("⚙️ Gestão de Limites de Alerta")
    st.info("Defina abaixo a partir de qual valor cada indicador é considerado ALERTA.")

    conn   = get_conn()
    df_lim = pd.read_sql("SELECT * FROM limites_exames", conn)
    conn.close()

    with st.form("form_limites"):
        novos_limites: dict[str, float] = {}
        for _, row in df_lim.iterrows():
            novos_limites[row["exame"]] = st.number_input(
                f"{row['exame'].capitalize()} ({row['unidade']})",
                value=float(row["valor_alerta"]),
                step=1.0,
            )
        salvar = st.form_submit_button("💾 Salvar Configurações", type="primary")

        if salvar:
            conn   = get_conn()
            cursor = conn.cursor()
            for exame, valor in novos_limites.items():
                cursor.execute(
                    "UPDATE limites_exames SET valor_alerta=? WHERE exame=?",
                    (valor, exame),
                )
            conn.commit()
            conn.close()
            st.success("✅ Limites atualizados!")


# ── Tela: Lançamento de Exames + Predição com IA ─────────────

def tela_predicao() -> None:
    st.title("🤖 Lançamento de Exames e Predição de Risco")

    conn      = get_conn()
    pacientes = pd.read_sql("SELECT id, nome FROM pacientes ORDER BY nome", conn)
    conn.close()

    if pacientes.empty:
        st.warning("Nenhum paciente cadastrado. Vá em 'Cadastro de Pacientes' primeiro.")
        return

    opcoes   = {row["nome"]: row["id"] for _, row in pacientes.iterrows()}
    nome_sel = st.selectbox("Selecione o Paciente", list(opcoes.keys()))
    id_pac   = opcoes[nome_sel]

    st.divider()
    st.subheader("📝 Inserir Valores dos Exames")

    col1, col2 = st.columns(2)
    with col1:
        glicose    = st.number_input("Glicose (mg/dL)",    min_value=0.0, max_value=500.0, value=100.0, step=1.0)
        pressao    = st.number_input("Pressão (mmHg)",     min_value=0.0, max_value=300.0, value=80.0,  step=1.0)
    with col2:
        imc        = st.number_input("IMC (kg/m²)",        min_value=0.0, max_value=80.0,  value=24.0,  step=0.1)
        colesterol = st.number_input("Colesterol (mg/dL)", min_value=0.0, max_value=600.0, value=180.0, step=1.0)

    if st.button("🔍 Analisar com IA", type="primary", use_container_width=True):
        model   = load_model()
        entrada = pd.DataFrame([{
            "glicose"   : glicose,
            "pressao"   : pressao,
            "imc"       : imc,
            "colesterol": colesterol,
        }])

        pred      = model.predict(entrada)[0]
        prob      = model.predict_proba(entrada)[0][1]
        resultado = "Alerta" if pred == 1 else "Normal"

        st.divider()

        if resultado == "Alerta":
            st.error(f"⚠️ RESULTADO: **RISCO ALTO (ALERTA)** — Probabilidade: {prob * 100:.1f}%")
            st.markdown(
                """
                <div style='background:#ffe0e0; padding:20px; border-radius:10px;
                            border-left:6px solid red;'>
                    <h3 style='color:red;'>⚠️ ATENÇÃO MÉDICA NECESSÁRIA</h3>
                    <p>O modelo identificou <strong>risco elevado</strong> para este paciente.
                    Recomenda-se avaliação clínica imediata.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.success(f"✅ RESULTADO: **NORMAL** — Probabilidade de risco: {prob * 100:.1f}%")
            st.markdown(
                """
                <div style='background:#e0ffe0; padding:20px; border-radius:10px;
                            border-left:6px solid green;'>
                    <h3 style='color:green;'>✅ Indicadores Dentro do Esperado</h3>
                    <p>Nenhum sinal de risco elevado detectado. Manter acompanhamento de rotina.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Persiste o resultado no banco
        conn   = get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO exames
               (id_paciente, glicose, pressao, imc, colesterol, resultado_ia, probabilidade)
               VALUES (?,?,?,?,?,?,?)""",
            (id_pac, glicose, pressao, imc, colesterol, resultado, round(prob, 4)),
        )
        conn.commit()
        conn.close()
        st.caption("📁 Resultado salvo no banco de dados.")

    # Histórico do paciente selecionado
    st.divider()
    st.subheader(f"📂 Histórico de Exames — {nome_sel}")

    conn = get_conn()
    hist = pd.read_sql(
        """
        SELECT glicose, pressao, imc, colesterol,
               resultado_ia, probabilidade, data_exame
        FROM exames
        WHERE id_paciente = ?
        ORDER BY id DESC
        LIMIT 20
        """,
        conn,
        params=(id_pac,),
    )
    conn.close()

    if hist.empty:
        st.info("Nenhum exame registrado para este paciente.")
    else:
        def colorir_resultado(val: str) -> str:
            if val == "Alerta":
                return "background-color:#ffcccc; color:#cc0000; font-weight:bold"
            return "background-color:#ccffcc; color:#006600; font-weight:bold"

        st.dataframe(
            hist.style.map(colorir_resultado, subset=["resultado_ia"]),
            use_container_width=True,
        )


# ── Roteador Principal ────────────────────────────────────────

def main() -> None:
    if "logado" not in st.session_state:
        st.session_state["logado"] = False

    # Bloqueia acesso se não estiver logado
    if not st.session_state["logado"]:
        tela_login()
        return

    # Sidebar de navegação
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=80)
        st.markdown(f"### 👤 {st.session_state['usuario']}")
        st.caption(f"Perfil: {st.session_state['perfil']}")
        st.divider()

        pagina = st.radio(
            "Navegação",
            [
                "📊 Dashboard",
                "👤 Cadastro de Pacientes",
                "⚙️ Gestão de Exames",
                "🤖 Análise com IA",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        if st.button("🚪 Sair", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    # Roteamento
    rotas = {
        "📊 Dashboard"            : tela_dashboard,
        "👤 Cadastro de Pacientes": tela_cadastro_pacientes,
        "⚙️ Gestão de Exames"     : tela_gestao_exames,
        "🤖 Análise com IA"       : tela_predicao,
    }
    rotas[pagina]()


if __name__ == "__main__":
    main()
