CREATE TABLE IF NOT EXISTS medicaments (
    cis                    BIGINT PRIMARY KEY,
    nom_medicament         TEXT NOT NULL,
    laboratoire            TEXT,
    code_atc               VARCHAR(20),
    molecule               TEXT,
    nb_patients_ville      FLOAT,
    substance_active       TEXT,
    pct_age_0_19_ans       FLOAT,
    pct_age_20_59_ans      FLOAT,
    pct_age_60_ans_et_plus FLOAT,
    pct_sexe_female        FLOAT,
    pct_sexe_male          FLOAT,
    est_antihistaminique   BOOLEAN
);

CREATE TABLE IF NOT EXISTS ruptures (
    id                   SERIAL PRIMARY KEY,
    cis                  BIGINT,
    nom_medicament       TEXT,
    cause                TEXT,
    classification       TEXT,
    date_evenement       DATE,
    code_atc             VARCHAR(20),
    molecule             TEXT,
    laboratoire          TEXT,
    nb_patients_ville    FLOAT,
    est_antihistaminique BOOLEAN,
    annee                INTEGER,
    mois                 INTEGER,
    trimestre            INTEGER,
    saison_allergies     BOOLEAN,
    cause_categorie      TEXT
);

CREATE TABLE IF NOT EXISTS bdpm (
    cis                  BIGINT PRIMARY KEY,
    atc                  VARCHAR(20),
    denomination         TEXT,
    lien                 TEXT,
    substance            TEXT,
    dosage               TEXT,
    est_antihistaminique BOOLEAN
);

CREATE TABLE IF NOT EXISTS meteo (
    id                   SERIAL PRIMARY KEY,
    time                 DATE,
    temperature_2m_max   FLOAT,
    temperature_2m_min   FLOAT,
    temperature_2m_mean  FLOAT,
    precipitation_sum    FLOAT,
    wind_speed_10m_max   FLOAT,
    aasqa                INTEGER,
    region               TEXT
);

CREATE TABLE IF NOT EXISTS openmedic (
    ATC1        VARCHAR(10),
    l_ATC1      TEXT,
    ATC2        VARCHAR(10),
    L_ATC2      TEXT,
    ATC3        VARCHAR(10),
    L_ATC3      TEXT,
    ATC4        VARCHAR(10),
    L_ATC4      TEXT,
    ATC5        VARCHAR(10),
    L_ATC5      TEXT,
    CIP13       BIGINT,
    l_cip13     TEXT,
    TOP_GEN     INTEGER,
    GEN_NUM     INTEGER,
    sexe        TEXT,
    BEN_REG     INTEGER,
    PSP_SPE     TEXT,
    BOITES      FLOAT,
    REM         TEXT,
    BSE         TEXT,
    annee       INTEGER,
    REM_clean   FLOAT,
    BSE_clean   FLOAT,
    AGE         TEXT
);

CREATE TABLE IF NOT EXISTS pollen (
    id          SERIAL PRIMARY KEY,
    date        DATE,
    region      TEXT,
    aasqa       INTEGER,
    taxon       TEXT,
    concentration FLOAT,
    niveau      TEXT,
    annee       INTEGER,
    mois        INTEGER,
    source      TEXT
);