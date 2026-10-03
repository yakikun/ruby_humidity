# frozen_string_literal: true
require 'minitest/autorun'
require_relative 'sensor_decision'

class SensorDecisionTest < Minitest::Test
  def base
    { 'air_temperature_c' => 31, 'relative_humidity_pct' => 50,
      'soil_moisture_pct' => 25, 'rain_probability_1h' => 10 }
  end

  def test_hot_dry_waters_more
    r = SensorDecision.evaluate(base)
    assert_equal 'water_hot', r['action']
    assert_equal 150.0, r['water_ml']
  end

  def test_rain_has_priority
    r = SensorDecision.evaluate(base.merge('rain_probability_1h' => 90))
    assert_equal 'hold_rain_expected', r['action']
    assert_equal 0.0, r['water_ml']
  end

  def test_wet_soil_holds
    r = SensorDecision.evaluate(base.merge('soil_moisture_pct' => 80))
    assert_equal 'hold_soil_wet', r['action']
  end

  def test_missing_and_invalid_values_rejected
    assert_raises(ArgumentError) { SensorDecision.evaluate(base.reject { |k, _| k == 'soil_moisture_pct' }) }
    assert_raises(ArgumentError) { SensorDecision.evaluate(base.merge('rain_probability_1h' => 'NaN')) }
    assert_raises(ArgumentError) { SensorDecision.evaluate(base.merge('relative_humidity_pct' => 101)) }
  end
end
