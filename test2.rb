require_relative 'lib/llama_rb'

ctx = Llama::Context.new
ctx.init!

prompt = "Building a website can be done in 10 simple steps: "
embd_inp = ctx.tokenize(prompt)

ctx.evaluate(embd_inp, 0, 4)
logits = ctx.logits

all_logits = 0.upto(logits.size - 1).flat_map do |row|
  0.upto(logits.n_cols - 1).map do |col|
    logits[row, col]
  end
end

binding.irb
