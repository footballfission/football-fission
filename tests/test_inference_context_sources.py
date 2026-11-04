"""One source schema must drive completion designs and reference domain checks."""
import unittest

from hierarchical_football.inference.baselines import passing_context_sources, passing_design
from hierarchical_football.inference.context_support import model_context_population
from tests.inference_fixtures import passing_frame


class PassingContextSources(unittest.TestCase):
    def test_existing_contexts_preserve_declared_sources_and_design_order(self):
        variants = {'linear': (), 'depth': (), 'position': ('position',), 'opponent': ('defteam',),
                    'depth_position': ('position',), 'depth_opponent': ('defteam',),
                    'position_opponent': ('position', 'defteam'), 'expanded': ('position', 'defteam')}
        table = passing_frame(dict(complete_pass=0), dict(air_yards=20.0))
        for context, extra in variants.items():
            with self.subTest(context=context):
                source = passing_context_sources(context)
                self.assertEqual(source['continuous'], ('air_yards', 'yardline_100', 'ydstogo'))
                self.assertEqual(source['categorical'], ('down', 'pass_location') + extra)
                design = passing_design(table, context=context)
                self.assertEqual(design.spec['categorical'], source['categorical'])
                self.assertEqual(design.spec['continuous'][:3], source['continuous'])
                audit = model_context_population({'fit': table}, table, context=context)['report']
                self.assertEqual(set(audit['windows']['fit']['bounds']), set(source['continuous']))
                self.assertEqual(set(audit['windows']['fit']['levels']), set(source['categorical']))
                self.assertEqual(audit['retained_rows'], table.height)

    def test_unknown_context_fails(self):
        with self.assertRaises(ValueError):
            passing_context_sources('unknown')


if __name__ == '__main__':
    unittest.main()
