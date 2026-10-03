# frozen_string_literal: true

# Conservative, explainable irrigation policy.
# It consumes hardware/forecast values; it does not pretend that a humidity
# prediction alone is an irrigation command.
module SensorDecision
  REQUIRED = %w[air_temperature_c relative_humidity_pct soil_moisture_pct rain_probability_1h].freeze
  module_function

  def finite(value, name)
    number = Float(value)
    raise ArgumentError, "#{name} must be finite" unless number.finite?
    number
  rescue ArgumentError, TypeError
    raise ArgumentError, "#{name} must be numeric"
  end

  def evaluate(reading, config = {})
    missing = REQUIRED.reject { |key| reading.key?(key) }
    raise ArgumentError, "missing sensor values: #{missing.join(', ')}" unless missing.empty?
    temp = finite(reading['air_temperature_c'], 'air_temperature_c')
    humidity = finite(reading['relative_humidity_pct'], 'relative_humidity_pct')
    soil = finite(reading['soil_moisture_pct'], 'soil_moisture_pct')
    rain = finite(reading['rain_probability_1h'], 'rain_probability_1h')
    raise ArgumentError, 'relative_humidity_pct must be 0..100' unless humidity.between?(0, 100)
    raise ArgumentError, 'soil_moisture_pct must be 0..100' unless soil.between?(0, 100)
    raise ArgumentError, 'rain_probability_1h must be 0..100' unless rain.between?(0, 100)
    cfg = {
      soil_dry_below_pct: 35.0,
      rain_hold_above_pct: 60.0,
      hot_above_c: 30.0,
      humidity_high_above_pct: 85.0,
      minimum_water_ml: 0.0,
      normal_water_ml: 100.0,
      hot_water_ml: 150.0
    }.merge(config.transform_keys(&:to_sym))
    reasons = []
    if rain >= cfg[:rain_hold_above_pct]
      action, water = 'hold_rain_expected', 0.0
      reasons << "1時間以内の降雨確率#{rain.round(1)}%が閾値以上"
    elsif soil >= 100.0 || soil >= 75.0
      action, water = 'hold_soil_wet', 0.0
      reasons << "土壌水分#{soil.round(1)}%が十分"
    elsif soil < cfg[:soil_dry_below_pct]
      action = temp >= cfg[:hot_above_c] ? 'water_hot' : 'water_normal'
      water = temp >= cfg[:hot_above_c] ? cfg[:hot_water_ml] : cfg[:normal_water_ml]
      reasons << "土壌水分#{soil.round(1)}%が乾燥閾値未満"
      reasons << "気温#{temp.round(1)}℃が高温閾値以上" if temp >= cfg[:hot_above_c]
    else
      action, water = 'hold_soil_not_dry', 0.0
      reasons << "土壌水分#{soil.round(1)}%は乾燥閾値以上"
    end
    { 'action' => action, 'water_ml' => water, 'reasons' => reasons,
      'inputs' => { 'air_temperature_c' => temp, 'relative_humidity_pct' => humidity,
                    'soil_moisture_pct' => soil, 'rain_probability_1h' => rain },
      'policy_version' => 'conservative_v0',
      'warning' => 'threshold policy; not validated for plant safety or automatic pump control' }
  end
end
