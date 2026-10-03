# frozen_string_literal: true
require_relative 'data_io'
raise ArgumentError, 'ruby predict.rb MODEL_JSON FEATURES_CSV' unless ARGV.length == 2
model, payload = HumidityData.load_model(ARGV[0])
_, rows = HumidityData.read(ARGV[1], feature_names: payload['feature_names'])
rows = HumidityData.transform(rows, payload['feature_mean'], payload['feature_std'])
rows.each do |r|
  puts JSON.generate(sample_id: r[:id], predicted_humidity_percent: model.predict_percent(r[:x]))
end
