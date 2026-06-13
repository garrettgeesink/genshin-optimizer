import { allSubstatKeys } from '@genshin-optimizer/gi/consts'
import type { ArtifactSlotKey, SubstatKey } from '@genshin-optimizer/gi/consts'
import type { ICachedArtifact } from '@genshin-optimizer/gi/db'
import type { IArtifact } from '@genshin-optimizer/gi/good'
import {
  deduplicate,
  dustReshape,
  expandNode,
  type MarkovNode,
  type ValuesLevelNode,
} from '@genshin-optimizer/gi/upopt'
import type { DynStat } from '@genshin-optimizer/gi/solver'
import { mvnPE_bad } from './mvncdf'
import { ResultType, type UpOptResult } from './upOpt'
import type { FullBuildEvaluator, ReshapeDisplayArtifact } from './types'

type WeightedNode = { p: number; n: MarkovNode }

export function buildReshapeArtifacts(
  artifacts: ICachedArtifact[],
  equippedArts: Record<ArtifactSlotKey, ICachedArtifact | undefined>,
  evalFullBuild: FullBuildEvaluator,
  thresholds: number[],
  mintotal: number,
): ReshapeDisplayArtifact[] {
  return artifacts
    .filter((art) => art.level === 20)
    .flatMap((art) =>
      getSubstatPairs(art).flatMap((affixes) => {
        try {
          return [
            createReshapeArtifact(
              art,
              equippedArts,
              evalFullBuild,
              thresholds,
              affixes,
              mintotal
            ),
          ]
        } catch {
          return []
        }
      })
    )
}

function createReshapeArtifact(
  art: ICachedArtifact,
  equippedArts: Record<ArtifactSlotKey, ICachedArtifact | undefined>,
  evalFullBuild: FullBuildEvaluator,
  thresholds: number[],
  affixes: [SubstatKey, SubstatKey],
  mintotal: number
): ReshapeDisplayArtifact {
  const buildNodes = (artifact: ICachedArtifact) =>
    dustReshape(toReshapeArtifact(artifact), equippedArts, affixes, mintotal)
  let nodes = buildNodes(art)
  const base = toBaseReshapeArtifact(art, affixes, mintotal)

  const out: ReshapeDisplayArtifact = {
    ...base,
    result: toFastResult(evalFullBuild, thresholds, nodes),
    recalc: (artifact) => {
      nodes = buildNodes(artifact)
      const next = toBaseReshapeArtifact(artifact, affixes, mintotal)
      Object.assign(out, next)
      out.result = toFastResult(evalFullBuild, thresholds, nodes)
    },
    calcExact: () => {
      if (out.result?.evalMode === ResultType.Exact) return
      out.result = toExactResult(evalFullBuild, thresholds, nodes)
    },
  }
  return out
}

function toBaseReshapeArtifact(
  art: ICachedArtifact,
  affixes: [SubstatKey, SubstatKey],
  mintotal: number
) {
  const subs = art.substats
    .map(({ key }) => key)
    .filter((key): key is SubstatKey => key !== '')
  const totalRolls =
    art.totalRolls ??
    art.substats.reduce((sum, substat) => sum + substat.rolls.length, 0)
  const values = {
    [art.setKey]: 1,
    [art.mainStatKey]: art.mainStatVal,
    ...Object.fromEntries(
      art.substats.flatMap((substat) => {
        if (!substat.key) return []
        const initialValue = substat.initialValue ?? substat.rolls[0]
        return initialValue === undefined
          ? []
          : [[substat.key, toDecimal(substat.key, initialValue)]]
      })
    ),
  } as DynStat
  return {
    key: `${art.id}:${affixes[0]}:${affixes[1]}:${mintotal}`,
    id: art.id,
    slotKey: art.slotKey,
    mainStat: art.mainStatKey,
    subs,
    values,
    rollsLeft: Math.max(0, totalRolls - subs.length),
    source: art,
    affixes,
    mintotal,
  }
}

function toReshapeArtifact(art: ICachedArtifact): IArtifact {
  return {
    setKey: art.setKey,
    rarity: art.rarity,
    level: art.level,
    slotKey: art.slotKey,
    mainStatKey: art.mainStatKey,
    substats: art.substats.map((substat) => {
      const initialValue = substat.initialValue ?? substat.rolls[0]
      return initialValue === undefined
        ? { key: substat.key, value: substat.value }
        : { key: substat.key, value: substat.value, initialValue }
    }),
    totalRolls:
      art.totalRolls ??
      art.substats.reduce((sum, substat) => sum + substat.rolls.length, 0),
    astralMark: art.astralMark,
    elixirCrafted: art.elixirCrafted,
    location: art.location,
    lock: art.lock,
    unactivatedSubstats: art.unactivatedSubstats?.map((substat) =>
      substat.initialValue === undefined
        ? { key: substat.key, value: substat.value }
        : {
            key: substat.key,
            value: substat.value,
            initialValue: substat.initialValue,
          }
    ),
  }
}

function getSubstatPairs(art: ICachedArtifact): [SubstatKey, SubstatKey][] {
  const subs = art.substats
    .map(({ key }) => key)
    .filter((key): key is SubstatKey => key !== '')
  const out: [SubstatKey, SubstatKey][] = []
  for (let i = 0; i < subs.length; i++) {
    for (let j = i + 1; j < subs.length; j++) out.push([subs[i], subs[j]])
  }
  return out
}

function toFastResult(
  evalFullBuild: FullBuildEvaluator,
  thresholds: number[],
  nodes: WeightedNode[]
): UpOptResult {
  let ptot = 0
  let upAvgtot = 0
  const gmm = nodes.map(({ p: phi, n }) => {
    const { prob, constr_prob, upAvg, f_mu, f_cov } = evaluateNode(
      evalFullBuild,
      thresholds,
      n
    )
    ptot += phi * prob
    upAvgtot += phi * prob * upAvg
    return { phi, cp: constr_prob, mu: f_mu[0], sig2: f_cov[0][0] }
  })
  const lowers = gmm.map(({ mu, sig2 }) => mu - 4 * Math.sqrt(sig2))
  const uppers = gmm.map(({ mu, sig2 }) => mu + 4 * Math.sqrt(sig2))
  return {
    p: ptot,
    upAvg: ptot < 1e-6 ? 0 : upAvgtot / ptot,
    distr: { gmm, lower: Math.min(...lowers), upper: Math.max(...uppers) },
    evalMode: ResultType.Fast,
  }
}

function toExactResult(
  evalFullBuild: FullBuildEvaluator,
  thresholds: number[],
  nodes: WeightedNode[]
): UpOptResult {
  let current = nodes
  const dummyObj = {
    computeWithDerivs: () => [[], [] as DynStat[]] as [number[], DynStat[]],
    threshold: [] as number[],
    zeroDeriv: [] as SubstatKey[],
  }
  while (current.some(({ n }) => n.type !== 'values')) {
    current = deduplicate(
      dummyObj,
      current.flatMap(({ p, n }) =>
        expandNode(n).map(({ p: childP, n: child }) => ({
          p: p * childP,
          n: child,
        }))
      )
    )
  }

  let ptot = 0
  let upAvgtot = 0
  const gmm = current.map(({ p: phi, n }) => {
    const values = evalFullBuild((n as ValuesLevelNode).subDistr.base).map(
      ({ v }) => v
    )
    const pass = values.every((value, i) => value >= thresholds[i])
    const cp = values
      .slice(1)
      .every((value, i) => value >= thresholds[i + 1])
    if (pass) {
      ptot += phi
      upAvgtot += phi * (values[0] - thresholds[0])
    }
    return { phi, cp: cp ? 1 : 0, mu: values[0], sig2: 0 }
  })
  const mus = gmm.map(({ mu }) => mu)
  return {
    p: ptot,
    upAvg: ptot < 1e-6 ? 0 : upAvgtot / ptot,
    distr: { gmm, lower: Math.min(...mus), upper: Math.max(...mus) },
    evalMode: ResultType.Exact,
  }
}

function evaluateNode(
  evalFullBuild: FullBuildEvaluator,
  thresholds: number[],
  node: MarkovNode
) {
  const { base, subs, mu, cov } = node.subDistr
  const stats = { ...base }
  subs.forEach((sub, i) => {
    stats[sub] = (stats[sub] ?? 0) + mu[i]
  })
  const objective = evalFullBuild(stats)
  const f_mu = objective.map(({ v }) => v)
  const f_cov = objective.map(({ grads }) =>
    objective.map(({ grads: grads2 }) =>
      subs.reduce(
        (sum, subA, ixA) =>
          sum +
          subs.reduce((inner, subB, ixB) => {
            const dfa = grads[allSubstatKeys.indexOf(subA)] ?? 0
            const dfb = grads2[allSubstatKeys.indexOf(subB)] ?? 0
            return inner + dfa * cov[ixA][ixB] * dfb
          }, 0),
        0
      )
    )
  )
  const { p: prob, upAvg, cp: constr_prob } = mvnPE_bad(
    f_mu,
    f_cov,
    thresholds
  )
  return { f_mu, f_cov, prob, constr_prob, upAvg }
}

function toDecimal(key: string, value: number) {
  return key.endsWith('_') ? value / 100 : value
}
