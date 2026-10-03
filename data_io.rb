# frozen_string_literal: true
require 'csv'
require_relative 'neural_network'

module HumidityData
  METADATA = %w[sample_id split humidity_pct].freeze
  module_function

  def number(value, context)
    raise ArgumentError, "#{context}: blank value" if value.nil? || value.to_s.strip.empty?
    begin
      parsed = Float(value)
    rescue ArgumentError, TypeError
      raise ArgumentError, "#{context}: invalid number"
    end
    raise ArgumentError, "#{context}: nonfinite value" unless parsed.finite?
    parsed
  end

  def read(path, training: false, feature_names: nil)
    table = CSV.read(path, headers: true, encoding: 'bom|utf-8')
    headers = table.headers
    unless headers.is_a?(Array) && !headers.empty? && headers.all? { |h| h.is_a?(String) && !h.strip.empty? } && headers.uniq == headers
      raise ArgumentError, 'empty or duplicate CSV headers'
    end
    raise ArgumentError, 'empty CSV dataset' if table.empty?
    found = headers - METADATA
    raise ArgumentError, 'no feature columns' if found.empty?
    names = feature_names || found
    unless found.sort == names.sort && names.uniq == names
      raise ArgumentError, "feature mismatch; missing=#{names - found}; unexpected=#{found - names}"
    end
    raise ArgumentError, 'training needs split and humidity_pct' if training && (%w[split humidity_pct] - headers).any?
    ids = {}
    rows = table.each_with_index.map do |row, i|
      raise ArgumentError, "row#{i + 2}: extra CSV cells" if row.fields.length != headers.length
      x = names.map { |name| number(row[name], "row#{i + 2}/#{name}") }
      id = headers.include?('sample_id') ? row['sample_id'] : "row#{i + 2}"
      raise ArgumentError, 'blank or duplicate sample_id' if id.nil? || id.strip.empty? || ids[id]
      ids[id] = true
      item = { id: id, x: x }
      if training
        label = number(row['humidity_pct'], "row#{i + 2}/humidity_pct")
        raise ArgumentError, 'humidity outside0to100' unless label.between?(0, 100)
        raise ArgumentError, 'split must be train/validation/diagnostic' unless %w[train validation diagnostic].include?(row['split'])
        item.merge!(y: label, split: row['split'])
      end
      item
    end
    [names, rows]
  end

  def fit_scaler(rows)
    raise ArgumentError, 'empty training data' if rows.empty?
    mean = rows.first[:x].each_index.map { |j| rows.sum { |r| r[:x][j] } / rows.length }
    std = mean.each_index.map { |j| [Math.sqrt(rows.sum { |r| (r[:x][j] - mean[j])**2 } / rows.length), 0.01].max }
    validate_scaler(mean, std, mean.length)
    [mean, std]
  end

  def validate_scaler(mean, std, n)
    unless mean.is_a?(Array) && std.is_a?(Array) && mean.length == n && std.length == n &&
           mean.all? { |v| HumidityMLP.finite_number?(v) } && std.all? { |v| HumidityMLP.finite_number?(v) && v.positive? }
      raise ArgumentError, 'invalid feature normalization'
    end
  end

  def transform(rows, mean, std)
    validate_scaler(mean, std, rows.first.fetch(:x).length)
    rows.map do |r|
      x = r[:x].each_with_index.map { |v, j| (v - mean[j]) / std[j] }
      raise ArgumentError, 'normalization overflow' unless x.all?(&:finite?)
      r.merge(x: x)
    end
  end

  def metrics(y, p)
    raise ArgumentError, 'nonempty matched targets/predictions required' if y.empty? || y.length != p.length
    errors = p.zip(y).map { |prediction, target| prediction - target }
    mse = errors.sum { |v| v**2 } / errors.length
    { n: errors.length, mae_rh_points: errors.sum(&:abs) / errors.length,
      mse_rh_points_squared: mse, rmse_rh_points: Math.sqrt(mse), mse_normalized: mse / 10_000.0 }
  end

  def load_model(path)
    payload = JSON.parse(File.read(path))
    raise ArgumentError, 'v2model with explicit schema required; legacy models are not silently guessed' unless payload['schema_version'] == 2
    names = payload.fetch('feature_names')
    model = HumidityMLP.from_state(payload.fetch('network'))
    unless names.is_a?(Array) && names.length == model.input_size && names.uniq == names && names.all? { |n| n.is_a?(String) && !n.empty? && !METADATA.include?(n) }
      raise ArgumentError, 'invalid feature schema'
    end
    validate_scaler(payload['feature_mean'], payload['feature_std'], model.input_size)
    [model, payload]
  end
end
