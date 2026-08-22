# -*- coding: utf-8 -*-
"""
Utilities for multistage stochastic PyPSA networks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import inspect
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pypsa2smspp.stochastic_utils
from pypsa2smspp.constants import STOCHASTIC_PARAMETER_REGISTRY

# =================================================
# Normalizzazione parametri stocastici
# Analogo di stochastic_utils.normalize_stochastic_parameters
# =================================================

def normalize_sddp_parameters(
        stochastic_parameters: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Normalizza i metadati stocastici forniti dall'utente per il caso SDDP.

    Atteso, analogamente al caso TSSB:
        {
            "stochastic_type": "sddp",
            "parameters": ["demand", "hydro_inflow", "renewable_maxpower", ...]
        }

    Riusa lo stesso STOCHASTIC_PARAMETER_REGISTRY di stochastic_utils.py:
    qualunque parametro già registrato per TSSB è automaticamente valido
    anche qui.
    TODO: controllare i parametri specifici per SDDP e, se mancanti, aggiungerli a STOCHASTIC_PARAMETER_REGISTRY
    """

    sp = dict(stochastic_parameters or {})
    stochastic_type = sp.get("stochastic_type", None)
    if stochastic_type != "sddp":
        raise ValueError(
            f"stochastic_type {stochastic_type} non valido."
            f"normalize_sddp_parameters richiede stochastic_type='sddp',"
            f"ricevuto {stochastic_type!r}."
        )

    parameters = sp.get("parameters", [])
    if parameters is None:
        parameters = []
    elif isinstance(parameters, str):
        parameters = [parameters]
    else:
        parameters = list(parameters)

    parameters = [str(p).strip().lower() for p in parameters if str(p).strip()]

    valid_parameters = set(STOCHASTIC_PARAMETER_REGISTRY)
    invalid = sorted(set(parameters) - valid_parameters)

    if invalid:
        raise ValueError(
            f"Parametri stocastici non supportati per SDDP: {invalid}. "
            f"Valori validi (STOCHASTIC_PARAMETER_REGISTRY): "
            f"{sorted(valid_parameters)}."
        )

    return {
        "stochastic_type": "sddp",
        "parameters": parameters,
    }

# =================================================
# Partizione in stadi?
# =================================================

# TODO: capire come gestire questo pezzo

# ================================================
# Estrazione dei dati per ogni singolo stadio
# ================================================

