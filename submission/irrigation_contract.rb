require 'time'
require 'digest'
require 'json'

# Review-only interface. This module cannot issue an executable pump command.
module IrrigationContract
  module_function
  def evaluate(reading, now: Time.now.utc)
    result = { schema_version: 1, mode: 'dry_run', executable: false,
               device_id: reading['device_id'], command: 'NO_ACTUATION',
               generated_at: now.iso8601, expires_at: (now + 60).iso8601,
               recommended_water_ml: nil, recommendation: 'review_required', reasons: [] }
    begin
      raise ArgumentError, 'missing_device_id' unless reading['device_id'].is_a?(String) && !reading['device_id'].strip.empty?
      stamp = reading.fetch('observed_at')
      raise ArgumentError, 'timezone_required' unless stamp.is_a?(String) && stamp.match?(/(Z|[+-]\d\d:\d\d)\z/)
      age = now - Time.iso8601(stamp)
      raise ArgumentError, 'stale_or_future_sensor_data' unless age.between?(0, 300)
      raise ArgumentError, 'soil_calibration_required' unless reading['soil_calibrated'] == true
      soil = reading.fetch('soil_moisture_pct')
      raise ArgumentError, 'invalid_soil_moisture' unless soil.is_a?(Numeric) && soil.finite? && soil.between?(0,100)
      raise ArgumentError, 'rain_observation_required' unless [true, false].include?(reading['raining_now'])
      # These are demo thresholds, not an approved horticultural policy.
      if reading['raining_now']
        result[:recommendation] = 'hold'; result[:reasons] << 'rain_observed'
      elsif soil >= 35
        result[:recommendation] = 'hold'; result[:reasons] << 'soil_not_dry_demo_threshold'
      else
        result[:recommendation] = 'watering_candidate'; result[:reasons] << 'soil_dry_demo_threshold'
      end
      result[:reasons] << 'water_volume_requires_plant_and_flow_calibration'
    rescue KeyError, ArgumentError => e
      result[:reasons] << e.message
    end
    result[:request_id] = Digest::SHA256.hexdigest(JSON.generate(result))[0,24]
    result
  end
end
