"""Tests for time handling logic in NS Agent ADK."""

import datetime
import unittest
from unittest import mock

from ns_agent_adk import engine as engine_module
from ns_agent_adk.config import config as config_module
from ns_agent_adk.core import fhir_time
from ns_agent_adk.core import graph as graph_module
from ns_agent_adk.core import linearizer as linearizer_module


class TestFHIRTimeExtraction(unittest.TestCase):

  def test_extract_encounter_time(self):
    resource = {
        'resourceType': 'Encounter',
        'id': 'enc1',
        'period': {
            'start': '2023-01-01T10:00:00Z',
            'end': '2023-01-01T12:00:00Z',
        },
    }
    ct = fhir_time.get_clinical_time(resource)
    self.assertIsNotNone(ct.range)
    self.assertEqual(
        ct.start_time,
        datetime.datetime(2023, 1, 1, 10, 0, 0, tzinfo=datetime.timezone.utc),
    )
    # probulates.start; production.end 是不提取的，所以
    # 范围.end is None and end_time 返回 start_time.
    self.assertIsNone(ct.range.end)
    self.assertEqual(ct.end_time, ct.start_time)

  def test_extract_observation_time_iso(self):
    resource = {
        'resourceType': 'Observation',
        'effectiveDateTime': '2023-05-20T14:30:00Z',
    }
    ct = fhir_time.get_clinical_time(resource)
    self.assertEqual(
        ct.start_time,
        datetime.datetime(2023, 5, 20, 14, 30, 0, tzinfo=datetime.timezone.utc),
    )

  def test_extract_partial_date(self):
    resource = {'resourceType': 'Condition', 'onsetDateTime': '2022-11'}
    ct = fhir_time.get_clinical_time(resource)
    self.assertEqual(
        ct.start_time,
        datetime.datetime(2022, 11, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
    )
    # 月底
    expected_end = datetime.datetime(
        2022, 11, 30, 23, 59, 59, 999999, tzinfo=datetime.timezone.utc
    )
    self.assertEqual(ct.end_time, expected_end)


class TestGraphConstruction(unittest.TestCase):

  def test_build_simple_bundle(self):
    bundle = {
        'entry': [
            {
                'resource': {
                    'resourceType': 'Encounter',
                    'id': 'enc1',
                    'period': {'start': '2023-01-01T08:00:00Z'},
                }
            },
            {
                'resource': {
                    'resourceType': 'Observation',
                    'id': 'obs1',
                    'effectiveDateTime': '2023-01-01T09:00:00Z',
                    'encounter': {'reference': 'Encounter/enc1'},
                }
            },
        ]
    }
    graph = graph_module.ChronologicalHypergraph()
    graph.build_from_bundle(bundle)

    self.assertEqual(len(graph.spine), 1)
    # 观测结果应该在交点上
    self.assertEqual(len(graph.spine[0].nodes), 1)
    self.assertEqual(graph.spine[0].nodes[0].id, 'obs1')


class TestAnchorTime(unittest.TestCase):

  def setUp(self):
    self.graph = graph_module.ChronologicalHypergraph()
    self.config = config_module.Config()
    # 嵌入器
    self.embedder_mock = mock.MagicMock()
    self.embedder_mock.embed.return_value = [[0.1, 0.2]]
    self.embedder_mock.similarity.return_value = 0.9
    self.config.get_embedder = mock.MagicMock(return_value=self.embedder_mock)
    self.config.get_llm_model = mock.MagicMock(return_value="test_model")

    # 添加一些节点
    node1 = graph_module.FHIRNode(
        id='obs1',
        resource_type='Observation',
        timestamp=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
        data={'code': {'text': 'Test Obs'}},
    )
    self.graph._node_map['obs1'] = node1
    # 返回此节点的模 get_all_nodes
    self.graph.get_all_nodes = mock.MagicMock(return_value=[node1])

  def test_linearizer_anchor_date_recency(self):
    # 未来定点日期方式 -> 节点是旧的 -> 低复用度？
    # 或 Anchor 日期接近 -> 节点是最近的 -> 高度复发性。

    anchor_date = datetime.date(2020, 1, 2)
    linearizer = linearizer_module.CausalSaliencyLinearizer(
        self.graph, self.config, anchor_date=anchor_date
    )

    # 移动时间解析器以返回“最近” 查询类型
    with mock.patch(
        'ns_agent_adk.core.linearizer.temporal_parser'
    ) as mock_tp:
      mock_tp.get_temporal_intent.return_value.query_type = 'recent'
      mock_tp.get_temporal_intent.return_value.anchor_event = None

      scores = linearizer.calculate_saliency('recent labs')
      # 三角洲是1天，静态分数应该很高。
      # score = semantic + temporal_mask + anchor_mask + recency * weight + always_include
      # 只要检查它运行时没有出错并返回分数
      self.assertIn('obs1', scores)

  def test_engine_pass_anchor_date(self):
    anchor_date = datetime.date(2025, 1, 1)
    with mock.patch(
        'ns_agent_adk.engine._load_preamble'
    ) as mock_preamble:
      agent = engine_module.NeuroSymbolicAgent(
          self.config, self.graph, anchor_date=anchor_date
      )
      # 检查前导是否装入 2025-01-01
      mock_preamble.assert_called_with(
          user_context=None,
          current_time=datetime.datetime(2025, 1, 1, 0, 0, tzinfo=datetime.timezone.utc),
          max_llm_calls=15,
          enabled_tools=self.config.enabled_tools,
      )


if __name__ == '__main__':
  unittest.main()
