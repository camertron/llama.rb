require_relative 'lib/llama_rb'
require 'etc'

# Llama::Context.prepend(
#   Module.new do
#     def init!
#       silence_stderr { super }
#     end

#     private

#     def silence_stderr
#       old_stream = STDERR.dup
#       STDERR.reopen(RbConfig::CONFIG['host_os'] =~ /mswin|mingw/ ? 'NUL:' : '/dev/null')
#       STDERR.sync = true
#       yield
#     ensure
#       STDERR.reopen(old_stream)
#     end
#   end
# )

ctx = Llama::Context.new
ctx.init!

prompt = "Building a website can be done in 10 simple steps: "
# prompt = "How do you get rid of pasty butt in chickens? "

# ctx.predict_each(prompt) do |token|
#   STDOUT.write(token)
# end

embd_inp = ctx.tokenize(prompt)
n_ctx = ctx.n_ctx
last_n_tokens = Array.new(n_ctx) { 0 }
newline = ctx.tokenize("\n", false)

n_predict     = 512
n_keep        = 0

n_past        = 0
n_remain      = n_predict
n_consumed    = 0
repeat_last_n = 64
n_batch       = 8

n_threads     = [4, Etc.nprocessors].min;
antiprompts   = []
interactive   = false

embd = ctx.make_token_list

# require "pry-byebug"
# binding.pry

while n_remain != 0
  # predict
  if embd.size > 0
    # infinite text generation via context swapping
    # if we run out of context:
    # - take the n_keep first tokens from the original prompt (via n_past)
    # - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch
    if n_past + embd.size > n_ctx
      n_left = n_past - n_keep
      n_past = n_keep

      # insert n_left / 2 tokens at the start of embd from last_n_tokens
      embd.unshift(
        last_n_tokens[(n_ctx - n_left / 2 - embd.size)..(last_n_tokens.size - embd.size)]
      )
    end

    unless ctx.evaluate(embd, n_past, n_threads)
      STDERR.puts("failed to eval")
      break
    end
  end

  n_past += embd.size
  embd.clear

  ignore_eos = false

  if embd_inp.size <= n_consumed
    # out of user input, sample next token
    top_k          = 40
    top_p          = 0.95
    temp           = 0.80
    repeat_penalty = 1.10

    id = 0
    logits = ctx.logits

    if ignore_eos
      logits[0, Llama::Token.eos] = 0
    end

    id = ctx.sample_top_p_top_k(
      last_n_tokens[(n_ctx - repeat_last_n)..-1],
      top_k,
      top_p,
      temp,
      repeat_penalty
    )

    last_n_tokens.shift
    last_n_tokens << id

    # replace end of text token with newline token when in interactive mode
    if id == Llama::Token.eos && interactive && !instruct
      id = newline[0].value;
      if antiprompts.size != 0
        # # tokenize and inject first reverse prompt
        # first_antiprompt = ctx.tokenize(antiprompts[0], false)
        # embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
      end
    end

    # add it to the context
    embd << id

    # decrement remaining sampling budget
    n_remain -= 1
  else
    # some user input remains from prompt or interaction, forward it to processing
    while embd_inp.size > n_consumed
      embd << embd_inp[n_consumed].value
      last_n_tokens.shift
      last_n_tokens << embd_inp[n_consumed].value
      n_consumed += 1

      if embd.size >= n_batch
        break
      end
    end
  end

  # display text
  embd.each do |token|
    STDOUT.write(token.to_s)
  end

  # end of text token
  if embd.last.value == Llama::Token.eos
    break
  end
end
