# -*- coding: utf-8 -*-
"""
Utilities for multistage stochastic PyPSA networks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pypsa2smspp.stochastic_utils as su
from pypsa2smspp.constants import STOCHASTIC_PARAMETER_REGISTRY

# =================================================
# Normalizzazione parametri stocastici
# Analogo di stochastic_utils.normalize_stochastic_parameters
# =================================================

def normalize_sddp_parameters(
        stochastic_parameters: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Normalizza e valida i parametri stocastici specifici per SDDP.

    Formato atteso in input (esempio):
        {
            "stochastic_type": "sddp",
            "parameters": ["demand", "renewable_maxpower"],
            "periods": [
                {"name": "2020", "snapshots": ["2020-01-01 00:00:00", ...]},
                {"name": "2021", "snapshots": ["2021-01-01 00:00:00", ...]},
            ]
        }

    La chiave "periods" è opzionale qui: se assente, verrà gestita altrove
    (es. usando `investment_periods` della rete). Se presente, deve avere
    una struttura valida.

    Returns
    -------
    dict
        Dizionario pulito con chiavi:
        - "stochastic_type": "sddp"
        - "parameters": lista di parametri stocastici validi
        - "periods": lista di dizionari periodi normalizzati (o None se non forniti)
    """

    sp = dict(stochastic_parameters or {})

    # Controllo tipo stocastico
    stochastic_type = sp.get("stochastic_type", None)
    if stochastic_type != "sddp":
        raise ValueError(
            f"stochastic_type {stochastic_type} non valido."
            f"normalize_sddp_parameters richiede stochastic_type='sddp',"
            f"ricevuto {stochastic_type!r}."
        )

    # Normalizzazione parametri stocastici
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

    # Validazione "periods" (opzionale)
    periods = sp.get("periods", None)

    if periods is not None:
        if not isinstance(periods, (list, tuple)) or len(periods) == 0:
            raise ValueError(
                "Il campo 'periods' deve essere una lista non vuota di "
                "dizionari con chiavi 'name' e 'snapshots'."
            )

        validated_periods = []
        seen_names = set()

        for idx, period in enumerate(periods):
            if not isinstance(period, dict):
                raise ValueError(
                    f"L'elemento {idx} di 'periods' non è un dizionario."
                )

            name = period.get("name", None)
            snapshots = period.get("snapshots", None)

            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"L'elemento {idx} di 'periods' deve avere un 'name' "
                    "stringa non vuoto."
                )

            if name in seen_names:
                raise ValueError(
                    f"Nome duplicato in 'periods': {name!r}."
                )
            seen_names.add(name)

            if snapshots is None or not isinstance(snapshots, (list, tuple)) or len(snapshots) == 0:
                raise ValueError(
                    f"Il periodo {name!r} deve avere una lista 'snapshots' "
                    "non vuota."
                )

            # Verifichiamo che snapshots siano omogenei (opzionale per ora).
            validated_periods.append({
                "name": name,
                "snapshots": list(snapshots),
            })

        periods = validated_periods

    return {
        "stochastic_type": "sddp",
        "parameters": parameters,
        "periods": periods,
    }



# =================================================
# Partizione in stadi
# =================================================

# TODO: capire come gestire questo pezzo

def get_sddp_stage_names(n, stochastic_parameters=None) -> List[Any]:
    """
    Restituisce la lista ordinata dei nomi degli stadi SDDP.

    La fonte degli stadi è, nell'ordine:
      1. Il campo "periods" di `stochastic_parameters` (se fornito).
      2. Altrimenti, `n.investment_periods` se presente.
      3. Altrimenti, errore.

    Questa funzione è importante per `transformation.py` perché determina
    quanti `StochasticBlock` creare nell'`SDDPBlock`.

    Parameters
    ----------
    n : pypsa.Network
        Rete PyPSA (possibilmente stocastica).
    stochastic_parameters : dict, optional
        Parametri stocastici già forniti dall'utente.

    Returns
    -------
    list
        Nomi degli stadi (es. ["2020", "2021", ...]).
    """
    # Normalizza e valida i parametri SDDP, ottenendo anche i periodi
    sp = normalize_sddp_parameters(stochastic_parameters)
    periods = sp.get("periods", None)

    # Caso 1: periodi forniti esplicitamente dall'utente
    if periods:
        return [p["name"] for p in periods]

    # Caso 2: fallback su investment_periods della rete
    investment_periods = getattr(n, "investment_periods", None)
    if investment_periods:
        return list(investment_periods)

    # Caso 3: nessuna informazione disponibile -> errore
    raise ValueError(
        "Per SDDP è necessario specificare 'periods' in stochastic_parameters "
        "oppure usare un PyPSA network con investment_periods."
    )

# ================================================
# Estrazione dei dati per ogni singolo stadio
# ================================================

def build_sddp_demand_stage(n, stage_snapshots: pd.Index) -> Dict[str, Any]:
    """
    Estrae i dati di domanda per un singolo stadio SDDP.

    Questa funzione è l'equivalente di `stochastic_utils.build_dss_demand`
    ma limitata agli snapshot di un solo stadio.

    Parametri
    ----------
    n : pypsa.Network
        Rete stocastica PyPSA.
    stage_snapshots : pd.Index o lista di timestamp
        Snapshot appartenenti a questo stadio.

    Ritorna
    -------
    dict
        Dizionario con:
        - "scenarios": matrice (NumberScenarios, SubScenarioSize_demand)
        - "pool_weights": probabilità scenario (vettore)
        - "node_order": ordine dei bus
        - "snapshot_order": ordine degli snapshot (dello stadio)
        - "scenario_size": dimensione del vettore di scenario per questo stadio
        - "number_scenarios": numero di scenari
        - altri metadati di flattening
    """
    scenario_names = su.get_scenario_names(n)
    if not scenario_names:
        raise ValueError(
            "build_sddp_demand_stage richiede una rete stocastica "
            "(n.has_scenarios)."
        )

    # Converti stage_snapshots in un pd.Index per operazioni di reindexing
    stage_snapshots = pd.Index(stage_snapshots)

    scenarios = []
    bus_order = None

    for scenario_name in scenario_names:
        # Prendi la rete per questo scenario
        n_s = n.get_scenario(scenario_name)

        # Estrai domanda per bus: DataFrame con righe=bus, colonne=snapshots
        demand_by_bus = su.get_bus_demand_matrix(n_s)

        # Verifica che tutti gli snapshot dello stadio siano presenti
        # Nota: se i tipi non coincidono, prova a convertire in DatetimeIndex
        if not all(s in demand_by_bus.columns for s in stage_snapshots):
            stage_snapshots_dt = pd.to_datetime(stage_snapshots)
            missing = [s for s in stage_snapshots_dt if s not in demand_by_bus.columns]
            if missing:
                raise ValueError(
                    f"Snapshot dello stadio non trovati nella domanda dello "
                    f"scenario {scenario_name!r}: {missing[:5]}"
                    f"{'...' if len(missing) > 5 else ''}"
                )
            # Se tutti trovati, aggiorna stage_snapshots alla versione Datetime
            stage_snapshots = stage_snapshots_dt

        # Filtra le colonne della domanda mantenendo solo gli snapshot dello stadio
        demand_by_bus = demand_by_bus.reindex(columns=stage_snapshots)

        # Salva l'ordine dei bus la prima volta e riordina le altre per coerenza
        if bus_order is None:
            bus_order = list(demand_by_bus.index)
        else:
            demand_by_bus = demand_by_bus.reindex(index=bus_order, fill_value=np.nan)

        # Appiattisci con ordine node_major_time_minor (come TSSB)
        scenarios.append(su.flatten_bus_demand_node_major(demand_by_bus))

    # Impila tutte le righe scenario in una matrice
    scenario_matrix = np.vstack(scenarios).astype(float)

    # Probabilità scenario (usate per dopo; di solito uniformi)
    pool_weights = su.get_scenario_probabilities(n).astype(float)

    return {
        "parameter": "demand",
        "scenarios": scenario_matrix,
        "pool_weights": pool_weights,
        "node_order": bus_order,
        "snapshot_order": list(stage_snapshots),
        "flattening": "node_major_time_minor",
        "scenario_size": int(scenario_matrix.shape[1]),
        "number_scenarios": int(scenario_matrix.shape[0]),
    }

def build_sddp_unitblock_timeseries_parameter_stage(
        n,
        stage_snapshots: pd.Index,
        parameter: str,
        pypsa_component: str,
        field: str,
        asset_names: Sequence[Any],
        function_name: str,
        unitblock_type: str,
        target: str,
        transformation_config,
        smspp_parameter: str | None,
        weights: bool,
)-> Dict[str, Any]:
    """
    Estrae dati di un parametro stocastico di tipo UnitBlock per un
    singolo stadio.

    Analogo a build_dss_unitblock_timeseries_parameter(...)
    """

    # Otteniamo i nomi degli scenari
    scenario_names = su.get_scenario_names(n);

    if not scenario_names:
        raise ValueError(
            f"build_sddp_unitblock_timeseries_parameter_stage richiede una rete stocastica (n.hash_scenarios)"
        )

    # Verifichiamo che asset_names non sia vuoto
    if not asset_names:
        raise ValueError(
            f"asset_names non può essere vuoto"
        )

    # Convertiamo stage_snapshots in pd.Index
    stage_snapshots = pd.Index(stage_snapshots)
    if not isinstance(stage_snapshots, pd.DatetimeIndex):
        stage_snapshots_dt = pd.to_datetime(stage_snapshots)
    else:
        stage_snapshots_dt = stage_snapshots

    # Inizializziamo i contenitori
    scenarios = []
    asset_order = None
    snapshot_order = None

    for scenario_name in scenario_names:
        # Otteniamo la rete dello scenario
        n_s = n.get_scenario(scenario_name)
        values = su.evaluate_unitblock_parameter_timeseries(
            n_s,
            parameter=parameter,
            pypsa_component=pypsa_component,
            field=field,
            asset_names=asset_names,
            transformation_config=transformation_config,
            unitblock_type=unitblock_type,
            smspp_parameter=smspp_parameter,
            weights=weights,
        )
        # Questa funzione restituisce un DataFrame con indice -> snapshot e colonne -> asset

        # Verifica che tutti gli snapshot dello stadio siano presenti
        missing = [s for s in stage_snapshots_dt if s not in values.index]
        if missing:
            raise ValueError(
                f"Snapshot dello stadio non trovati per {parameter!r} "
                f"(scenario {scenario_name!r}): {missing[:5]}"
            )

        # Filtra per gli snapshot dello stadio
        values = values.reindex(index=stage_snapshots_dt)

        if asset_order is None:
            asset_order = list(values.columns)
            snapshot_order = list(values.index)
        else:
            values = values.reindex(
                index=snapshot_order,
                columns=asset_order,
            )

        # Controllo valori mancanti
        if values.isna().any().any():
            missing_assets = values.columns[values.isna().any(axis=0)].tolist()
            raise ValueError(
                f"Valori mancanti per {parameter!r} in questo stadio. "
                f"Asset coinvolti: {missing_assets}"
            )

        scenarios.append(su.flatten_asset_timeseries_asset_major(values))

    # Costruiamo matrice finale
    scenario_matrix = np.vstack(scenarios).astype(float)
    # Otteniamo pool_weights
    pool_weights = su.get_scenario_probabilities(n).astype(float)

    return {
        "parameter": parameter,
        "target": target,
        "function_name": function_name,
        "unitblock_type": unitblock_type,
        "smspp_parameter": smspp_parameter,
        "pypsa_component": pypsa_component,
        "field": field,
        "source": f"{pypsa_component}.{field}",
        "scenarios": scenario_matrix,
        "pool_weights": pool_weights,
        "asset_order": asset_order,
        "snapshot_order": snapshot_order,
        "flattening": "asset_major_time_minor",
        "scenario_size": int(scenario_matrix.shape[1]),
        "number_scenarios": int(scenario_matrix.shape[0]),
    }

def build_sddp_stage_data(
        n,
        stage_name,
        stage_snapshots: pd.Index,
        stochastic_parameters: Sequence[str],
        intermittent_carriers,
        default_intermittent_carriers,
        enable_thermal_units: bool,
        transformation_config,
) -> Dict[str, Any]:
    """
    Per un singolo stadio, estrae i dati di tutti i parametri stocastici
    richiesti e li raggruppa in una lista parts
    """
    parts = []
    for parameter in stochastic_parameters:
        spec = STOCHASTIC_PARAMETER_REGISTRY[parameter]
        mapping_kind = spec["mapping_kind"]
        if mapping_kind == "ucblock_timeseries":
            if parameter != "demand":
                raise NotImplementedError("Per il momento supportato solo demand")
            parts.append(build_sddp_demand_stage(n, stage_snapshots))

        elif mapping_kind == "unitblock_timeseries":
            asset_names = su.get_stochastic_parameter_asset_names(
                n=n,
                parameter=parameter,
                spec=spec,
                intermittent_carriers=intermittent_carriers,
                default_intermittent_carriers=default_intermittent_carriers,
                enable_thermal_units=enable_thermal_units,
            )
            parts.append(
                build_sddp_unitblock_timeseries_parameter_stage(
                    n,
                    stage_snapshots,
                    parameter=parameter,
                    pypsa_component=spec["pypsa_component"],
                    field=spec["field"],
                    asset_names=asset_names,
                    function_name=spec["function_name"],
                    unitblock_type=spec["unitblock_type"],
                    target=spec["target"],
                    transformation_config=transformation_config,
                    smspp_parameter=spec.get("smspp_parameter", None),
                    weights=bool(spec.get("weights", False)),
                )
            )
        else:
            raise ValueError("mapping_kind deve essere supportato")

    return {
        "stage": stage_name,
        "parts": parts
    }


def merge_sddp_stage_data(stage_parts, stage_name=None) -> Dict[str, Any]:
    """
    Data una lista di parti stocastiche di un singolo stadio, produce il vettore
    di scenario completo per quello stadio e le informazioni necessarie per
    il costruttore di SDDPBlock.

    Parametri
    ----------
    stage_parts : list of dict
        Lista di dizionari, ognuno dei quali è il risultato di
        build_sddp_demand_stage o build_sddp_unitblock_timeseries_parameter_stage.
    stage_name : str, optional
        Nome dello stadio; se fornito viene incluso nell'output.

    Ritorna
    -------
    dict
        Dizionario con le chiavi:
        - "stage": nome stadio (solo se stage_name non è None)
        - "scenarios": matrice (NumberScenarios, SubScenarioSize)
        - "sub_scenario_size": somma delle dimensioni delle parti
        - "size_random_data_groups": lista delle dimensioni di ciascun gruppo
        - "pool_weights": vettore delle probabilità scenario
        - "parts": copia di stage_parts con offset_start/offset_end aggiunti
    """
    if not stage_parts:
        raise ValueError("stage_parts non può essere vuoto")

    # Il numero di scenari deve essere lo stesso per tutte le parti
    number_scenarios = stage_parts[0]["number_scenarios"]
    if number_scenarios == 0:
        raise ValueError("number_scenarios non può essere zero")

    # Lista per raccogliere gli array di scenario di ogni parte
    scenario_arrays = []
    # Lista delle dimensioni dei gruppi (una per parte)
    size_random_data_groups = []
    # Riferimento ai pool_weights (vettore probabilità)
    pool_weights = None

    # Lista per le parti con offset calcolati
    checked_parts = []
    offset = 0

    for part in stage_parts:
        # Verifica coerenza del numero di scenari
        if int(part["number_scenarios"]) != number_scenarios:
            raise ValueError(
                f"Tutte le parti devono avere lo stesso number_scenarios. "
                f"Atteso {number_scenarios}, ricevuto {part['number_scenarios']} "
                f"per la parte {part.get('parameter')!r}."
            )

        # Verifica coerenza delle probabilità scenario
        part_pool_weights = np.asarray(part["pool_weights"], dtype=float)
        if pool_weights is None:
            pool_weights = part_pool_weights
        elif not np.allclose(part_pool_weights, pool_weights):
            raise ValueError(
                "Tutte le parti devono avere gli stessi pool_weights. "
                f"Mismatch trovato per la parte {part.get('parameter')!r}."
            )

        # Estrai la matrice degli scenari come array float
        scenarios = np.asarray(part["scenarios"], dtype=float)
        part_size = scenarios.shape[1]

        # Aggiungi alla lista degli array
        scenario_arrays.append(scenarios)
        # Registra la dimensione del gruppo
        size_random_data_groups.append(part_size)

        # Crea una copia della parte con offset start/end
        part_copy = dict(part)
        part_copy["offset_start"] = offset
        part_copy["offset_end"] = offset + part_size
        checked_parts.append(part_copy)

        # Aggiorna l'offset per la prossima parte
        offset += part_size

        # Concatenazione orizzontale di tutte le parti
    stage_scenarios = np.hstack(scenario_arrays)  # shape (NumberScenarios, SubScenarioSize)
    sub_scenario_size = int(stage_scenarios.shape[1])

    # Costruzione del dizionario di output
    result = {
        "scenarios": stage_scenarios,
        "sub_scenario_size": sub_scenario_size,
        "size_random_data_groups": size_random_data_groups,
        "pool_weights": pool_weights,
        "parts": checked_parts,
    }

    # Includi il nome dello stadio solo se fornito
    if stage_name is not None:
        result["stage"] = stage_name

    return result

def build_sddp_scenarios(stage_data_list: list[dict]) -> Dict[str, Any]:
    """
    Concatena i vettori di ogni stadio
    in una matrice unica (NumberScenarios, ScenarioSize)
    """
    if not stage_data_list:
        raise ValueError("stage_data_list non è conforme")

    ref_stage = stage_data_list[0]
    sub_scenario_size_ref = ref_stage["sub_scenario_size"]
    pool_weights_ref = np.asarray(ref_stage["pool_weights"], dtype=float)
    number_scenarios_ref = ref_stage["scenarios"].shape[0]

    for stage in stage_data_list:
        if stage["sub_scenario_size"] != sub_scenario_size_ref:
            raise ValueError(
                "Tutti gli stadi devono avere lo stesso SubScenarioSize."
            )

        stage_pool_weights = np.asarray(stage["pool_weights"], dtype=float)
        if not np.allclose(stage_pool_weights, pool_weights_ref):
            raise ValueError(
                "Tutti gli stadi devono avere gli stessi pool_weights."
            )

        if stage["scenarios"].shape[0] != number_scenarios_ref:
            raise ValueError(
                "Tutti gli stadi devono avere lo stesso number_scenarios."
            )

    stage_scenario_list = [stage["scenarios"] for stage in stage_data_list]
    scenarios = np.hstack(stage_scenario_list)
    scenario_size = scenarios.shape[1]

    return {
        "scenarios": scenarios,
        "number_scenarios": number_scenarios_ref,
        "scenario_size": scenario_size,
        "sub_scenario_size": sub_scenario_size_ref,
        "pool_weights": pool_weights_ref,
        # opzionale:
        "stage_names": [stage.get("stage") for stage in stage_data_list]
    }


def build_sddp_dimensions(
        stage_data_list,
        scenarios_info,
        state_info = None,
        num_sub_blocks_per_stage=1,
) -> Dict[str, Any]:
    """
    Calcola le dimensioni necessarie per il costruttore di SDDPBlock
    """
    time_horizon = len(stage_data_list)

    # Estraiamo informazioni da scenarios_info
    sub_scenario_size = scenarios_info["sub_scenario_size"]
    number_scenarios = scenarios_info["number_scenarios"]
    scenario_size = scenarios_info["scenario_size"]

    # Dimensione dei random data groups (costante tra stadi)
    size_random_data_groups = stage_data_list[0]["size_random_data_groups"]
    num_random_data_groups = len(size_random_data_groups)

    # Gestione variabili di stato
    if state_info is None:
        state_size = 0
        initial_state = np.array([], dtype = float)
        admissible_state = np.array([], dtype = float)
        admissible_state_size = 0
        initial_state_size = 0
    else:
        state_size = state_info["state_size"]
        initial_state = np.asarray(state_info["initial_state"], dtype=float)
        admissible_state = np.asarray(state_info["admissible_state"], dtype=float)
        initial_state_size = initial_state.size  # o len(initial_state) se 1D
        admissible_state_size = admissible_state.size

    return {
    "TimeHorizon": time_horizon,
    "NumSubBlocksPerStage": num_sub_blocks_per_stage,
    "NumberScenarios": number_scenarios,
    "ScenarioSize": scenario_size,
    "SubScenarioSize": sub_scenario_size,
    "NumberRandomDataGroups": num_random_data_groups,
    "SizeRandomDataGroups": size_random_data_groups,
    "AdmissibleStateSize": admissible_state_size,
    "InitialStateSize": initial_state_size,
    "StateSize": state_size,
    "InitialState": initial_state,
    "AdmissibleState": admissible_state,
}