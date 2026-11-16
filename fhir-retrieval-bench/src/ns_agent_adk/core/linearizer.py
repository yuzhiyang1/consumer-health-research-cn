# src/fhir_neuro_symbolic/core/linearizer.py
"""Module for linearizing a ChronologicalHypergraph based on query saliency."""

import collections
import datetime
from unittest import mock

from absl import logging
import numpy as np

from ns_agent_adk.config import config as config_module
from ns_agent_adk.core import graph as graph_module
from ns_agent_adk.utils import embeddings as embeddings_module


try:
  from ns_agent_adk.utils import temporal_parser
except ImportError:
  temporal_parser = None


def _serialize_node_for_embedding(node: graph_module.FHIRNode) -> str:
  """Creates a value-aware string representation of a node for embedding."""
  data = node.data
  res_type = node.resource_type
  parts = [f'Type: {res_type}']

  try:
    if res_type == 'Observation':
      code = data.get('code', {}).get('text', 'N/A')
      value = data.get('valueQuantity', {}).get('value', 'N/A')
      unit = data.get('valueQuantity', {}).get('unit', '')
      time = node.timestamp.isoformat() if node.timestamp else 'N/A'
      parts.extend(
          [f'Code: {code}', f'Value: {value} {unit}'.strip(), f'Time: {time}']
      )
    elif res_type == 'Condition':
      code = data.get('code', {}).get('text', 'N/A')
      status = (
          data.get('clinicalStatus', {})
          .get('coding', [{}])[0]
          .get('code', 'N/A')
      )
      onset = node.timestamp.isoformat() if node.timestamp else 'N/A'
      parts.extend(
          [f'Condition: {code}', f'Status: {status}', f'Onset: {onset}']
      )
    elif res_type == 'Procedure':
      code = data.get('code', {}).get('text', 'N/A')
      time = node.timestamp.isoformat() if node.timestamp else 'N/A'
      parts.extend([f'Procedure: {code}', f'Time: {time}'])
    elif res_type == 'Encounter':
      enc_class = data.get('class', {}).get('code', 'N/A')
      start = node.timestamp.isoformat() if node.timestamp else 'N/A'
      end = data.get('period', {}).get('end', 'N/A')
      parts.extend(
          [f'Encounter Class: {enc_class}', f'Start: {start}', f'End: {end}']
      )
    else:
      # 通用后退
      name = data.get('code', {}).get('text', data.get('id', 'N/A'))
      time = node.timestamp.isoformat() if node.timestamp else 'N/A'
      parts.extend([f'Name: {name}', f'Time: {time}'])
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f'Error serializing node {node.id}: {e}')
    parts.append('Serialization Error')

  return ', '.join(parts)


class CausalSaliencyLinearizer:
  """Linearizes a graph into a text prompt based on causal saliency."""

  def __init__(
      self,
      graph: graph_module.ChronologicalHypergraph,
      config: config_module.Config,
      node_embeddings: dict[str, np.ndarray] | None = None,
      anchor_date: datetime.date | datetime.datetime | None = None,
  ):
    """Initializes the linearizer."""
    self.graph = graph
    self.config = config
    self.embedder = config.get_embedder()
    self.node_embeddings = node_embeddings
    self.anchor_date = anchor_date

  def calculate_saliency(self, query: str) -> dict[str, float]:
    """Calculates saliency scores for all nodes against the query."""
    query_embedding = self.embedder.embed([query])[0]
    saliency_scores: dict[str, float] = {}

    nodes_to_embed = self.graph.get_all_nodes()
    if not nodes_to_embed:
      return saliency_scores

    # 确定max_timestamp 时间意图提取
    max_timestamp = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    if self.anchor_date:
      if isinstance(self.anchor_date, datetime.datetime):
        max_timestamp = self.anchor_date
      else:
        # 转换日期至日期
        max_timestamp = datetime.datetime.combine(
            self.anchor_date, datetime.time.min
        ).replace(tzinfo=datetime.timezone.utc)
    else:
      # 倒转到图最大逻辑
      for node in nodes_to_embed:
        if node.timestamp and node.timestamp > max_timestamp:
          max_timestamp = node.timestamp

    time_range = mock.MagicMock()
    time_range.query_type = 'none'
    time_range.anchor_event = None
    if temporal_parser:
      try:
        time_range = temporal_parser.get_temporal_intent(query, config=self.config, anchor_date=max_timestamp)
      except Exception:  # pylint: disable=broad-except
        pass

    # 嵌入节点
    if self.node_embeddings is not None:
      node_embeddings_dict: dict[str, np.ndarray] = self.node_embeddings
      node_embeddings = np.array([
          node_embeddings_dict.get(node.id, np.zeros(self.embedder.vector_size))
          for node in nodes_to_embed
      ])
    else:
      node_texts = [
          _serialize_node_for_embedding(node) for node in nodes_to_embed
      ]
      node_embeddings = self.embedder.embed(node_texts)
      if not hasattr(self.embedder, 'vector_size') and node_embeddings.size > 0:
        self.embedder.vector_size = node_embeddings[0].shape[0]

    # 解析Anchor事件(如"手术后").
    anchor_time = None
    if time_range.anchor_event:
      anchor_embedding = self.embedder.embed([time_range.anchor_event])[0]
      best_score = -1.0
      for i, node in enumerate(nodes_to_embed):
        score = self.embedder.similarity(anchor_embedding, node_embeddings[i])
        if score > best_score:
          best_score = score
          anchor_time = node.timestamp

    for i, node in enumerate(nodes_to_embed):
      node_embedding = node_embeddings[i]
      semantic_similarity = self.embedder.similarity(
          query_embedding, node_embedding
      )

      # 1. 临时蒙面(距离)
      temporal_mask = 1.0
      if node.timestamp and time_range.query_type not in ('none', 'recent'):
        node_ts = node.timestamp
        if time_range.start and node_ts < time_range.start:
          temporal_mask = 0.1
        if time_range.end and node_ts > time_range.end:
          temporal_mask = 0.1

      # 2 . 静态助推
      recency_score = 0.0
      if (
          node.timestamp
          and time_range.query_type == 'recent'
          and max_timestamp
          > datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
      ):
        # 基于max_timestamp天数的线性衰变(可能是anchor_date)
        delta_days = (max_timestamp - node.timestamp).days
        # 衰竭于1年(365天)
        recency_score = max(0.0, 1.0 - (delta_days / 365.0))

      # 3. 相对时间(锁定).
      anchor_mask = 1.0
      if anchor_time and node.timestamp:
        if time_range.raw_query_span and 'after' in time_range.raw_query_span:
          if node.timestamp < anchor_time:
            anchor_mask = 0.1
        elif (
            time_range.raw_query_span and 'before' in time_range.raw_query_span
        ):
          if node.timestamp > anchor_time:
            anchor_mask = 0.1

      # 4。总是包含
      always_include_boost = 0.0
      if node.resource_type in self.config.always_include_types:
        always_include_boost = 1.0

      # 组合分数
      # 基数:语义学* 面具
      final_score = semantic_similarity * temporal_mask * anchor_mask
      # 添加脚步
      final_score += recency_score * self.config.recency_weight
      final_score += always_include_boost

      saliency_scores[node.id] = final_score

    return saliency_scores

  def _format_hypernode(
      self,
      hn: graph_module.Hypernode,
      salient_nodes: list[graph_module.FHIRNode],
  ) -> str:
    """Formats a hypernode and its salient nodes into a string."""
    if not hn.start_time:
      return ''
    hn_start = hn.start_time.strftime('%Y-%m-%d')
    hn_end = hn.end_time.strftime('%Y-%m-%d') if hn.end_time else hn_start
    header = (
        f'=== Episode: {hn.id} ({hn_start} to {hn_end})'
        f" {'(Phantom)' if hn.is_phantom else ''} ==="
    )
    lines = [header]
    # 按时间戳排序突出节点
    salient_nodes.sort(
        key=lambda x: x.timestamp
        or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    )
    for node in salient_nodes:
      lines.append(f'  - {_serialize_node_for_embedding(node)}')
    return '\n'.join(lines)

  def _format_gap(
      self, start_time: datetime.datetime, end_time: datetime.datetime
  ) -> str:
    """Formats a time gap into a human-readable string."""
    delta = end_time - start_time
    if delta.days <= 0:
      return ''

    if delta.days > 365:
      gap = (
          f'{delta.days // 365} Years {delta.days % 365 // 30} Months'
          f' {(delta.days % 365) % 30} Days'
      )
    elif delta.days > 30:
      gap = f'{delta.days // 30} Months {delta.days % 30} Days'
    else:
      gap = f'{delta.days} Days'

    return (
        f"... [Skipped {gap} from {start_time.strftime('%Y-%m-%d')} to"
        f" {end_time.strftime('%Y-%m-%d')}] ..."
    )

  def linearize(self, query: str) -> str:
    """Linearizes the graph into a skeleton string based on strategy."""
    strategy = getattr(self.config, 'linearization_strategy', 'greedy')

    hn_candidates = []

    # 1.根据请求的战略算法分割执行路径
    if strategy == 'greedy':
      # --- PATH A: Saliency / Relevance Packed (Greedy) ---
      saliency_scores = self.calculate_saliency(query)

      for hn in self.graph.spine:
        spine_node = self.graph.get_node_by_id(hn.id)
        nodes_in_hn = hn.nodes + (
            [spine_node] if spine_node and spine_node not in hn.nodes else []
        )

        # 达到显著阈值的过滤节点
        salient_nodes = []
        hn_max_score = 0.0
        for node in nodes_in_hn:
          score = saliency_scores.get(node.id, 0.0)
          if score >= self.config.saliency_threshold:
            salient_nodes.append(node)
          hn_max_score = max(hn_max_score, score)

        if hn_max_score >= self.config.saliency_threshold:
          hn_text = self._format_hypernode(hn, salient_nodes)
          hn_tokens = len(hn_text.split())
          hn_candidates.append({
              'hn': hn,
              'score': hn_max_score,
              'text': hn_text,
              'tokens': hn_tokens,
              'start_time': hn.start_time,
          })

      # 最高功用等级
      hn_candidates.sort(key=lambda x: x['score'], reverse=True)

    elif strategy == 'chronological':
      # --- PATH B: Chronological Sequential Packed (Bypasses Saliency) ---
      for hn in self.graph.spine:
        spine_node = self.graph.get_node_by_id(hn.id)
        nodes_in_hn = hn.nodes + (
            [spine_node] if spine_node and spine_node not in hn.nodes else []
        )

        if not nodes_in_hn:
          continue

        # 直接包装完成超节点而不过滤高空
        hn_text = self._format_hypernode(hn, nodes_in_hn)
        hn_tokens = len(hn_text.split())
        hn_candidates.append({
            'hn': hn,
            'score': 1.0,  # 结构均匀性模拟分数
            'text': hn_text,
            'tokens': hn_tokens,
            'start_time': hn.start_time,
        })

      # 按最新时间顺序排列，首先确定预算最近活动的优先次序
      hn_candidates.sort(
          key=lambda x: x.get('start_time')
          or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc),
          reverse=True,
      )
    else:
      raise ValueError(f'Unknown linearization strategy: {strategy}')

    # 2. 常见逻辑流:将候选人打包入线性预算。
    selected_hns = []
    current_tokens = 0
    for cand in hn_candidates:
      if (
          current_tokens + cand['tokens']
      ) <= self.config.max_linearization_tokens:
        selected_hns.append(cand)
        current_tokens += cand['tokens']
      else:
        # 停止或跳过不再适合的候选者
        continue

    # 3. 常见逻辑流:将最终选择重新放回自然时间顺序
    selected_hns.sort(
        key=lambda x: x.get('start_time')
        or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    )

    # 5. 带有漏洞的格式
    output: list[str] = []
    last_salient_hn_end_time = None

    for cand in selected_hns:
      hn = cand['hn']
      if last_salient_hn_end_time and hn.start_time:
        gap_str = self._format_gap(last_salient_hn_end_time, hn.start_time)
        if gap_str:
          output.append(gap_str)

      output.append(cand['text'])
      last_salient_hn_end_time = hn.end_time

    if not output:
      return 'No relevant information found.'

    return '\n\n'.join(output)


def precompute_node_embeddings(
    graph: graph_module.ChronologicalHypergraph,
    embedder: embeddings_module.VectorCompass,
    cache_path: str | None = None,
) -> dict[str, np.ndarray]:
  """Precomputes embeddings for all nodes in the graph.

  This function serializes each node in the provided graph, computes their
  embeddings using the given embedder, and caches these embeddings. It can
  optionally load and save embeddings from/to a cache file to speed up
  subsequent calls.

  Args:
    graph: The ChronologicalHypergraph containing the nodes to embed.
    embedder: An instance of VectorCompass used to compute embeddings.
    cache_path: An optional file path to load/save the embedding cache.
      (`.parquet` format is supported.)

  Returns:
    A dictionary mapping node IDs to their corresponding embeddings (numpy
    arrays).

  Raises:
    ValueError: If inconsistent vector sizes are found in the precomputed
      embeddings or if they don't match the embedder's expected size.
  """
  nodes = graph.get_all_nodes()
  if not nodes:
    return {}
  
  if not cache_path:
    embedding_pool = {}
  else:
    embedding_pool = embeddings_module.load_embedding_pool(cache_path)
    if embedding_pool:
      logging.info('Loaded %d embeddings from cache.', len(embedding_pool))

  # 为所有节点准备文本数据，并识别缺失嵌入。
  # 必要时更新embeddding_pool
  texts_not_in_cache = collections.OrderedDict()
  for node in nodes:
    text = _serialize_node_for_embedding(node)
    text_hash = embeddings_module.get_cache_hash(text, embedder.model_name)

    if text_hash not in embedding_pool:
      texts_not_in_cache[text_hash] = text
  if texts_not_in_cache:
    embeddings_module.update_embedding_pool(embedder, embedding_pool, texts_not_in_cache)
    if cache_path:
      embeddings_module.save_embedding_pool(embedding_pool, cache_path)

  # 构造最终结果映射。
  node_embeddings = {
      node.id: embedding_pool[embeddings_module.get_cache_hash(_serialize_node_for_embedding(node), embedder.model_name)]
      for node in nodes
  }
 
  # 验证嵌入维度的一致性。
  if node_embeddings:
    embedding_dims = {vec.shape[0] for vec in node_embeddings.values()}
    if len(embedding_dims) > 1:
      raise ValueError(f'Inconsistent vector sizes found: {embedding_dims}')  
    current_dim = list(embedding_dims)[0]
    if current_dim != embedder.vector_size:
      raise ValueError(
          f'Inconsistent vector sizes found: computed: {current_dim} vs'
          f' embedder.vector_size: {embedder.vector_size}'
      )

  return node_embeddings
