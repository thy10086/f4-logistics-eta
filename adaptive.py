"""Allocate delivery promises to risk groups under both F4 carrier limits."""

import argparse
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path

import duckdb
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.stats import CensoredData, ecdf

import matched
import release as v2
import solution as core


MIN_SUPPORT = 300
ONTIME_OPTIONS = (.84, .88, .90, .93, .955)
TAIL_OPTIONS = (.92, .95, .965, .975, .985, .991, .996)
OPTIONS = tuple(product(ONTIME_OPTIONS, TAIL_OPTIONS))
SCENARIOS = tuple(product((.072, .084, .093), (.012, .018, .024)))


@dataclass
class Curve:
    knots: np.ndarray
    mass: np.ndarray

    def cdf(self, points):
        index = np.searchsorted(self.knots, points, side="right") - 1
        return np.where(index < 0, 0.0, self.mass[np.maximum(0, index)])

    def quantile(self, probability):
        index = np.searchsorted(self.mass, probability - 1e-10)
        return float(self.knots[min(index, len(self.knots) - 1)])


def distribution_axes(batch):
    location = batch["q90"]
    scale = np.clip(batch["q97"] - location, 16, 210)
    return location, scale


def fit_curve(base, scale, truth, known, elapsed):
    outcomes = (truth[known] - base[known]) / scale[known]
    under_observation = (elapsed[~known] - base[~known]) / scale[~known]
    if not len(outcomes):
        raise ValueError("A delay distribution needs known outcomes")
    data = CensoredData(uncensored=outcomes, right=under_observation) if len(under_observation) else outcomes
    curve = ecdf(data).cdf
    return Curve(curve.quantiles, curve.probabilities)


def segments(loc, scale, width_edges, center_edge):
    return 2 * np.searchsorted(width_edges, scale, side="right") + (loc > center_edge)


def hours_at(location, scale, minimum, action, padding=0):
    on, tail = action
    return np.maximum.reduce([location + scale * on, location + scale * tail - 24, minimum]) + padding


@dataclass
class Part:
    segment: int
    size: int
    actions: np.ndarray
    results: np.ndarray


@dataclass
class Carrier:
    group: int
    width_edges: np.ndarray
    center_edge: float
    parts: list[Part]


def reference_rows(batch, cutoff, min_age):
    if cutoff is None:
        active = np.ones(len(batch["y"]), dtype=bool)
        signed = active
        elapsed = np.zeros(active.sum())
    else:
        active = batch["pay_day"] < cutoff - min_age
        signed = batch["sign_day"][active] < cutoff
        elapsed = cutoff * 24 - batch["pay_hour"][active]
    return {key: values[active] for key, values in batch.items()}, signed, elapsed


def measure_actions(location, scale, floor, size, distribution, options):
    # Monotonic candidate actions retain their interpretation across shifts in mix.
    measurements = []
    for action in options:
        hours = hours_at(location, scale, floor, action)
        miss = 1 - distribution.cdf((hours - location) / scale)
        tail = 1 - distribution.cdf((hours + 24 - location) / scale)
        measurements.append((np.mean(hours), np.mean(miss), np.mean(tail)))
    return Part(0, size, options, np.array(measurements))


def train(reference, cutoff, min_age):
    history, signed, elapsed = reference_rows(reference, cutoff, min_age)
    loc, scale = distribution_axes(history)
    floor = history["floor"]
    result = {}
    for group in np.unique(history["carrier"]):
        in_group = history["carrier"] == group
        if np.count_nonzero(signed & in_group) < MIN_SUPPORT:
            continue
        width_edges = np.unique(np.quantile(scale[in_group], (.25, .50, .75)))
        center_edge = float(np.median(loc[in_group]))
        membership = segments(loc, scale, width_edges, center_edge)
        common = fit_curve(loc[in_group], scale[in_group], history["y"][in_group], signed[in_group], elapsed[in_group])
        parts = []
        for segment in np.unique(membership[in_group]):
            mask = in_group & (membership == segment)
            distribution = common
            if np.count_nonzero(signed & mask) >= MIN_SUPPORT:
                distribution = fit_curve(loc[mask], scale[mask], history["y"][mask], signed[mask], elapsed[mask])
            levels = {p: distribution.quantile(p) for p in set(ONTIME_OPTIONS + TAIL_OPTIONS)}
            unique_actions = np.unique([(levels[p_on], levels[p_tail]) for p_on, p_tail in OPTIONS], axis=0)
            part = measure_actions(loc[mask], scale[mask], floor[mask], int(mask.sum()), distribution, unique_actions)
            part.segment = int(segment)
            parts.append(part)
        result[int(group)] = Carrier(int(group), width_edges, center_edge, parts)
    return result


def select(carrier, risk_on, risk_tail):
    counts = np.array([len(part.actions) for part in carrier.parts])
    choices = np.repeat(np.arange(len(counts)), counts)
    total = sum(part.size for part in carrier.parts)
    proportions = np.repeat([part.size / total for part in carrier.parts], counts)
    outcomes = np.concatenate([part.results for part in carrier.parts]) * proportions[:, None]
    groups = (np.arange(len(counts))[:, None] == choices).astype(float)
    solver = milp(
        c=outcomes[:, 0], integrality=np.ones(len(choices)), bounds=Bounds(0, 1),
        constraints=[LinearConstraint(groups, 1, 1), LinearConstraint(outcomes[:, 1:].T, 0, [risk_on, risk_tail])],
        options={"node_limit": 10000, "mip_rel_gap": 0.002},
    )
    # A bound may be unattainable on a sparsely observed right tail.
    if solver.x is None or np.any(outcomes[:, 1:].T @ (solver.x > .5) > [risk_on + 1e-8, risk_tail + 1e-8]):
        return {}
    selected = np.flatnonzero(solver.x > .5)
    if not np.array_equal(choices[selected], np.arange(len(counts))):
        return {}
    start = np.r_[0, np.cumsum(counts[:-1])]
    return {part.segment: part.actions[index] for part, index in zip(carrier.parts, selected - start, strict=True)}


def prediction(reference, target, cutoff, policies, limits, fallback=None, padding=0):
    hours = v2.promises(reference, target, cutoff) if fallback is None else fallback.copy()
    location, scale = distribution_axes(target)
    solved = set()
    for group, policy in policies.items():
        actions = select(policy, *limits)
        if not actions:
            continue
        same_carrier = target["carrier"] == group
        membership = segments(location[same_carrier], scale[same_carrier], policy.width_edges, policy.center_edge)
        rows = np.flatnonzero(same_carrier)
        for segment, action in actions.items():
            use = rows[membership == segment]
            hours[use] = hours_at(location[use], scale[use], target["floor"][use], action, padding)
        solved.add(group)
    return core.finalize(hours, target["floor"]), solved


def fast_score(target, hours):
    carrier, ids = np.unique(target["carrier"], return_inverse=True)
    sizes = np.bincount(ids)
    mismatch = target["y"] - hours
    on = np.bincount(ids, weights=mismatch <= 0) / sizes
    tail = np.bincount(ids, weights=mismatch > 24) / sizes
    me = abs(float(mismatch.mean()))
    on_pen = float(500 * np.maximum(0, .9 - on).sum())
    tail_pen = float(500 * np.maximum(0, tail - .03).sum())
    return dict(A=me + on_pen + tail_pen, B=me + on_pen, ME=me, on=int((on >= .9).sum()),
                severe=int((tail <= .03).sum()), on_pen=on_pen, severe_pen=tail_pen,
                max_severe=float(tail.max()), avg_hours=float(hours.mean()),
                carriers=list(map(int, carrier)))


def study(cache, path):
    dates = (("stress", matched.STRESS_START), ("hold", matched.HOLD_START))
    grids = {}
    for name, date in dates:
        ref, target, _ = v2.read_batches(cache / f"{name}.npz")
        cutoff = matched.day(date)
        fallback = v2.promises(ref, target, cutoff)
        for age in (12, 17):
            policies = train(ref, cutoff, age)
            for limits in SCENARIOS:
                hours, solved = prediction(ref, target, cutoff, policies, limits, fallback)
                for padding in (0, 4, 8, 12, 16):
                    proposed = np.where(np.isin(target["carrier"], list(solved)), hours + padding, hours)
                    tag = f"age{age}_{limits[0]}_{limits[1]}_pad{padding}"
                    grids.setdefault(tag, {})[name] = dict(**fast_score(target, proposed), solved_groups=sorted(solved))
            core.log(f"Finished {name}, age {age}")
    path.write_text(json.dumps(grids, indent=2), encoding="utf-8")
    qualified = [tag for tag, row in grids.items() if all(item["on"] == item["severe"] == 8 for item in row.values())]
    for label, selection in [("qualified", qualified), ("all", list(grids))]:
        best = sorted(selection, key=lambda tag: grids[tag]["hold"]["A"])
        for tag in best[:8]:
            core.log(f"{label}: {tag}: {grids[tag]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--study", required=True, type=Path)
    args = parser.parse_args()
    study(args.cache, args.study)


if __name__ == "__main__":
    main()
