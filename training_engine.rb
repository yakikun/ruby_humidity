# frozen_string_literal: true
require_relative 'data_io'

module HumidityTrainer
  module_function
  def evaluate(model, rows)
    HumidityData.metrics(rows.map { |r| r[:y] }, rows.map { |r| model.predict_percent(r[:x]) })
  end

  # Diagnostic rows cannot influence fitting: they are not an argument here.
  def fit(model, train, validation, epochs: 80, learning_rate: 0.03, seed: 42, patience: 10)
    raise ArgumentError, 'train and validation must be nonempty' if train.empty? || validation.empty?
    raise ArgumentError, 'invalid epochs/patience' unless epochs.is_a?(Integer) && epochs.positive? && patience.is_a?(Integer) && patience >= 0
    rng = Random.new(seed)
    history = []; best = Float::INFINITY; best_state = nil; best_epoch = nil; stale = 0
    epochs.times do |i|
      online = (0...train.length).to_a.shuffle(random: rng).sum do |j|
        model.train_one(train[j][:x], train[j][:y], learning_rate: learning_rate)
      end / train.length
      tm = evaluate(model, train); vm = evaluate(model, validation)
      raise ArgumentError, 'nonfinite training metric' unless [tm, vm].all? { |m| m.values.all?(&:finite?) }
      history << { epoch: i + 1, online_train_mse_normalized: online, train: tm, validation: vm }
      if vm[:mae_rh_points] < best
        best = vm[:mae_rh_points]; best_state = model.state; best_epoch = i + 1; stale = 0
      else
        stale += 1
      end
      break if patience.positive? && stale >= patience
    end
    { model: HumidityMLP.from_state(best_state), last_state: model.state,
      history: history, best_epoch: best_epoch, epochs_completed: history.length }
  end
end
