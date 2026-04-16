"""
=============================================================
SCRIPT 03 – BACKEND / API DE PREDIÇÃO
Sistema de Predição de Risco Clínico – Documentação v1.0
=============================================================
Referência: Seção 2 – Arquitetura (Backend: FastAPI)
Referência: Seção 3 – Casos de uso (inserir dados,
            consultar risco, histórico, relatório)
Referência: Seção 4 – Funcionalidades
Referência: Seção 6 – Modelo de Dados
=============================================================
Execução:
    pip install fastapi uvicorn joblib scikit-learn pandas
    uvicorn 03_api_backend:app --reload --port 8000

Endpoints:
    POST /prever          → predição de risco (caso de uso 1+3)
    GET  /historico       → lista todos os registros (caso de uso 5)
    GET  /paciente/{id}   → busca paciente por ID (caso de uso 2)
    PUT  /paciente/{id}   → atualiza dados (caso de uso 4)
    GET  /relatorio       → estatísticas gerais (caso de uso 6)
    GET  /docs            → Swagger UI automático
=============================================================
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import joblib
import datetime
import uuid

# ─────────────────────────────────────────────────────────
# MODELOS PYDANTIC (Seção 6 – Modelo de Dados)
# ─────────────────────────────────────────────────────────

class DadosPaciente(BaseModel):
    """Tabela Paciente – Seção 6"""
    nome:             str   = Field(..., example="Ana")
    idade:            int   = Field(..., ge=18, le=99, example=45)
    glicose:          float = Field(..., ge=60,  le=300, example=105)
    pressao_arterial: float = Field(..., ge=80,  le=200, example=120)
    imc:              float = Field(..., ge=16,  le=50,  example=27.5)
    colesterol:       float = Field(..., ge=120, le=350, example=210)

class ResultadoPaciente(BaseModel):
    """Tabela Resultado – Seção 6"""
    paciente_id:   str
    nome:          str
    risco:         str          # "baixo" | "médio" | "alto"
    risco_codigo:  int          # 0 | 1 | 2
    probabilidade: dict         # {baixo, medio, alto}
    data_consulta: str

class AtualizacaoPaciente(BaseModel):
    """Campos opcionais para atualização – Seção 4 item 5"""
    glicose:          Optional[float] = None
    pressao_arterial: Optional[float] = None
    imc:              Optional[float] = None
    colesterol:       Optional[float] = None

# ─────────────────────────────────────────────────────────
# INICIALIZAÇÃO
# ─────────────────────────────────────────────────────────

app = FastAPI(
    title="API – Predição de Risco Clínico",
    description=(
        "Backend do Sistema de Predição de Risco Clínico. "
        "Arquitetura em 3 camadas – Seção 2 da Documentação Técnica."
    ),
    version="1.0.0"
)

# Carrega modelo treinado (Seção 7 – Random Forest)
try:
    modelo = joblib.load("modelo_risco_clinico.pkl")
except FileNotFoundError:
    modelo = None   # rode 02_pipeline_ml.py primeiro

FEATURES    = ["idade", "glicose", "pressao_arterial", "imc", "colesterol"]
LABEL_MAP   = {0: "baixo", 1: "médio", 2: "alto"}
historico_db: dict = {}   # Simula banco de dados em memória

# ─────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────

@app.get("/", tags=["Status"])
def status():
    """Health-check da API"""
    return {
        "sistema": "Predição de Risco Clínico",
        "versao":  "1.0.0",
        "modelo_carregado": modelo is not None
    }


@app.post("/prever", response_model=ResultadoPaciente, tags=["Predição"])
def prever_risco(dados: DadosPaciente):
    """
    Caso de uso 1+3 (Seção 3): Inserir dados do paciente e
    consultar risco clínico.
    Seção 5: normaliza, retorna probabilidade + classificação.
    """
    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute 02_pipeline_ml.py primeiro."
        )

    # Prepara entrada (Seção 5: normalização feita pelo pipeline)
    X = pd.DataFrame([[
        dados.idade, dados.glicose,
        dados.pressao_arterial, dados.imc, dados.colesterol
    ]], columns=FEATURES)

    # Predição (Seção 7)
    classe = int(modelo.predict(X)[0])
    probs  = modelo.predict_proba(X)[0]

    resultado = ResultadoPaciente(
        paciente_id   = str(uuid.uuid4())[:8],
        nome          = dados.nome,
        risco         = LABEL_MAP[classe],
        risco_codigo  = classe,
        probabilidade = {
            "baixo": round(float(probs[0]), 4),
            "medio": round(float(probs[1]), 4),
            "alto":  round(float(probs[2]), 4)
        },
        data_consulta = datetime.datetime.now().isoformat()
    )

    # Persiste no histórico (Seção 5: armazenar histórico)
    historico_db[resultado.paciente_id] = {
        **dados.dict(),
        **resultado.dict()
    }

    return resultado


@app.get("/historico", tags=["Histórico"])
def listar_historico():
    """
    Caso de uso: Visualizar histórico de pacientes (Seção 3).
    """
    if not historico_db:
        return {"total": 0, "registros": []}
    return {
        "total":     len(historico_db),
        "registros": list(historico_db.values())
    }


@app.get("/paciente/{paciente_id}", tags=["Histórico"])
def buscar_paciente(paciente_id: str):
    """
    Caso de uso: Consultar risco clínico de paciente específico (Seção 3).
    """
    if paciente_id not in historico_db:
        raise HTTPException(status_code=404, detail="Paciente não encontrado.")
    return historico_db[paciente_id]


@app.put("/paciente/{paciente_id}", tags=["Atualização"])
def atualizar_paciente(paciente_id: str, dados: AtualizacaoPaciente):
    """
    Caso de uso: Atualizar dados do paciente (Seção 3 + Seção 4 item 5).
    Re-processa a predição com os novos dados.
    """
    if paciente_id not in historico_db:
        raise HTTPException(status_code=404, detail="Paciente não encontrado.")

    # Atualiza apenas os campos enviados
    registro = historico_db[paciente_id]
    update = dados.dict(exclude_none=True)
    registro.update(update)

    # Re-predição com dados atualizados
    if modelo:
        X = pd.DataFrame([[
            registro["idade"], registro["glicose"],
            registro["pressao_arterial"], registro["imc"],
            registro["colesterol"]
        ]], columns=FEATURES)
        classe = int(modelo.predict(X)[0])
        probs  = modelo.predict_proba(X)[0]
        registro["risco"]         = LABEL_MAP[classe]
        registro["risco_codigo"]  = classe
        registro["probabilidade"] = {
            "baixo": round(float(probs[0]), 4),
            "medio": round(float(probs[1]), 4),
            "alto":  round(float(probs[2]), 4)
        }
        registro["data_consulta"] = datetime.datetime.now().isoformat()

    historico_db[paciente_id] = registro
    return {"mensagem": "Dados atualizados.", "registro": registro}


@app.get("/relatorio", tags=["Relatório"])
def gerar_relatorio():
    """
    Caso de uso: Gerar relatório (Seção 3 + Seção 4 item 6).
    Retorna estatísticas consolidadas do histórico.
    """
    if not historico_db:
        return {"mensagem": "Nenhum registro no histórico."}

    df = pd.DataFrame(historico_db.values())
    dist = df["risco"].value_counts().to_dict()

    return {
        "total_consultas": len(df),
        "distribuicao_risco": dist,
        "media_idade":      round(df["idade"].mean(), 1),
        "media_glicose":    round(df["glicose"].mean(), 1),
        "media_pa":         round(df["pressao_arterial"].mean(), 1),
        "media_imc":        round(df["imc"].mean(), 2),
        "media_colesterol": round(df["colesterol"].mean(), 1),
        "data_relatorio":   datetime.datetime.now().isoformat()
    }
