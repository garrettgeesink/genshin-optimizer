import type {
  ArtifactSlotKey,
  MainStatKey,
  SubstatKey,
} from '@genshin-optimizer/gi/consts'
import type { ICachedArtifact } from '@genshin-optimizer/gi/db'
import type { DynStat } from '@genshin-optimizer/gi/solver'
import type { UpOptResult } from './upOpt'

export type FullBuildEvaluator = (
  stats: DynStat
) => { v: number; grads: number[] }[]

export type ReshapeDisplayArtifact = {
  key: string
  id: string
  slotKey: ArtifactSlotKey
  mainStat: MainStatKey
  subs: SubstatKey[]
  values: DynStat
  rollsLeft: number
  source: ICachedArtifact
  affixes: [SubstatKey, SubstatKey]
  mintotal: number
  result?: UpOptResult
  recalc: (art: ICachedArtifact) => void
  calcExact: () => void
}
