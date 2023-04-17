#include <stdio.h>
#include <unordered_map>
#include <iostream>
#include <ruby.h>
#include "llama-cpp/examples/common.h"
#include "llama-cpp/llama.h"

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#include <signal.h>
#endif

static VALUE llama_rb_context_class;
static VALUE llama_rb_token_class;
static VALUE llama_rb_token_list_class;
static VALUE llama_rb_logits_class;

struct llama_rb_context {
    struct llama_context *ctx;
    int last_n_tokens;
};

void llama_rb_context_free(void* _context) {
    struct llama_rb_context *context = (struct llama_rb_context*)_context;

    if (context->ctx == NULL) {
        return;
    }

    llama_free(context->ctx);
    context->ctx = NULL;
}

void llama_rb_context_mark(void* _context) {
}

size_t llama_rb_context_size(const void* _context) {
	return sizeof(struct llama_rb_context);
}

static const rb_data_type_t llama_rb_context_type = {
	.wrap_struct_name = "llama_rb_context",
	.function = {
        .dmark = llama_rb_context_mark,
        .dfree = llama_rb_context_free,
        .dsize = llama_rb_context_size,
	},
	.flags = RUBY_TYPED_FREE_IMMEDIATELY,
};

struct llama_rb_token_list {
    struct llama_context *ctx;
    std::vector<llama_token> *tokens;
    std::unordered_map<int, VALUE> *ruby_tokens;
};

void llama_rb_token_list_free(void* _token_list) {
    struct llama_rb_token_list *token_list = (struct llama_rb_token_list*)_token_list;

    if (token_list->tokens != NULL) {
        token_list->tokens->clear();
        delete token_list->tokens;
        token_list->tokens = NULL;
    }

    if (token_list->ruby_tokens != NULL) {
        token_list->ruby_tokens->clear();
        delete token_list->ruby_tokens;
        token_list->ruby_tokens = NULL;
    }
}

void llama_rb_token_list_mark(void* _token_list) {
    struct llama_rb_token_list *token_list = (struct llama_rb_token_list*)_token_list;

    if (token_list->ruby_tokens == NULL) return;

    for(auto it = token_list->ruby_tokens->begin(); it != token_list->ruby_tokens->end(); it++) {
        rb_gc_mark(it->second);
    }
}

size_t llama_rb_token_list_size(const void* token_list) {
	return sizeof(struct llama_rb_token_list);
}

static const rb_data_type_t llama_rb_token_list_type = {
	.wrap_struct_name = "llama_rb_token_list",
	.function = {
        .dmark = llama_rb_token_list_mark,
        .dfree = llama_rb_token_list_free,
        .dsize = llama_rb_token_list_size,
	},
	.flags = RUBY_TYPED_FREE_IMMEDIATELY,
};

struct llama_rb_token {
    struct llama_context *ctx;
    llama_token token;
};

void llama_rb_token_free(void* _token) {
}

void llama_rb_token_mark(void* _token) {
}

size_t llama_rb_token_size(const void* _token) {
	return sizeof(struct llama_rb_token);
}

static const rb_data_type_t llama_rb_token_type = {
	.wrap_struct_name = "llama_rb_token",
	.function = {
        .dmark = llama_rb_token_mark,
        .dfree = llama_rb_token_free,
        .dsize = llama_rb_token_size,
	},
	.flags = RUBY_TYPED_FREE_IMMEDIATELY,
};

struct llama_rb_logits {
    float *logits;
    int n_tokens;
    int n_vocab;
};

void llama_rb_logits_free(void* _logits) {
}

void llama_rb_logits_mark(void* _logits) {
}

size_t llama_rb_logits_size(const void* _logits) {
	return sizeof(struct llama_rb_logits);
}

static const rb_data_type_t llama_rb_logits_type = {
	.wrap_struct_name = "llama_rb_logits",
	.function = {
        .dmark = llama_rb_logits_mark,
        .dfree = llama_rb_logits_free,
        .dsize = llama_rb_logits_size,
	},
	.flags = RUBY_TYPED_FREE_IMMEDIATELY,
};

VALUE llama_rb_context_alloc(VALUE self) {
    struct llama_rb_context *context;
    context = (struct llama_rb_context *)malloc(sizeof(struct llama_rb_context));
    context->ctx = NULL;
    context->last_n_tokens = 0;

    return TypedData_Wrap_Struct(self, &llama_rb_context_type, context);
}

VALUE llama_rb_context_init(VALUE self) {
    struct llama_rb_context *context;
    TypedData_Get_Struct(self, struct llama_rb_context, &llama_rb_context_type, context);

    struct llama_context_params lparams = llama_context_default_params();
    const char* model = "/Users/camertron/Downloads/llama/7B/ggml-model-q4_0.bin";

    context->ctx = llama_init_from_file(model, lparams);

    if (context->ctx == NULL) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, model);
        return Qnil;
    } else {
        fprintf(stderr, "Model loaded!");
    }

    return Qnil;
}

VALUE llama_rb_context_tokenize(int argc, VALUE* argv, VALUE self) {
    VALUE text, add_bos;
    rb_scan_args(argc, argv, "11", &text, &add_bos);

    if (NIL_P(add_bos)) {
        add_bos = Qtrue;
    }

    struct llama_rb_context *context;
    TypedData_Get_Struct(self, struct llama_rb_context, &llama_rb_context_type, context);

    std::vector<llama_token> res(RSTRING_LEN(text) + 1);
    int n = llama_tokenize(context->ctx, StringValuePtr(text), res.data(), res.size(), add_bos == Qtrue);

    res.resize(n);

    struct llama_rb_token_list *list;
    list = (struct llama_rb_token_list *)malloc(sizeof(struct llama_rb_token_list));
    list->ctx = context->ctx;
    list->tokens = new std::vector<llama_token>(n);
    list->ruby_tokens = NULL;

    for (size_t i = 0; i < n; i++) {
        list->tokens->at(i) = res[i];
    }

    return TypedData_Wrap_Struct(llama_rb_token_list_class, &llama_rb_token_list_type, list);
}

VALUE llama_rb_context_n_ctx(VALUE self) {
    struct llama_rb_context *context;
    TypedData_Get_Struct(self, struct llama_rb_context, &llama_rb_context_type, context);

    return INT2FIX(llama_n_ctx(context->ctx));
}

VALUE llama_rb_context_make_token_list(int argc, VALUE* argv, VALUE self) {
    struct llama_rb_context *context;
    TypedData_Get_Struct(self, struct llama_rb_context, &llama_rb_context_type, context);

    VALUE tokens;
    rb_scan_args(argc, argv, "01", &tokens);

    int n = 0;

    if (!NIL_P(tokens)) {
        n = rb_array_len(tokens);
    }

    struct llama_rb_token_list *list;
    list = (struct llama_rb_token_list *)malloc(sizeof(struct llama_rb_token_list));
    list->ctx = context->ctx;
    list->tokens = new std::vector<llama_token>(n);
    list->ruby_tokens = NULL;

    if (!NIL_P(tokens)) {
        for (int i = 0; i < n; i ++) {
            list->tokens->at(i) = FIX2LONG(rb_ary_entry(tokens, i));
        }
    }

    return TypedData_Wrap_Struct(llama_rb_token_list_class, &llama_rb_token_list_type, list);
}

VALUE llama_rb_context_evaluate(VALUE self, VALUE _token_list, VALUE n_past, VALUE n_threads) {
    struct llama_rb_context *context;
    TypedData_Get_Struct(self, struct llama_rb_context, &llama_rb_context_type, context);

    struct llama_rb_token_list *token_list;
    TypedData_Get_Struct(_token_list, struct llama_rb_token_list, &llama_rb_token_list_type, token_list);

    int result = llama_eval(
        context->ctx,
        token_list->tokens->data(),
        token_list->tokens->size(),
        FIX2INT(n_past),
        FIX2INT(n_threads)
    );

    context->last_n_tokens = token_list->tokens->size();

    if (result == 0) {
        return Qtrue;
    } else {
        return Qfalse;
    }
}

VALUE llama_rb_token_list_entry_at(VALUE self, long idx) {
    struct llama_rb_token_list *list;
    TypedData_Get_Struct(self, struct llama_rb_token_list, &llama_rb_token_list_type, list);

    if (idx < 0) {
        idx = list->tokens->size() + idx;
    }

    if (idx < 0 || idx >= list->tokens->size()) return Qnil;

    if (list->ruby_tokens == NULL) {
        list->ruby_tokens = new std::unordered_map<int, VALUE>();
    }

    auto token = list->ruby_tokens->find(idx);

    if (token == list->ruby_tokens->end()) {
        struct llama_rb_token *new_token;
        new_token = (struct llama_rb_token *)malloc(sizeof(struct llama_rb_token));
        new_token->ctx = list->ctx;
        new_token->token = list->tokens->at(idx);
        auto new_rtoken = TypedData_Wrap_Struct(llama_rb_token_class, &llama_rb_token_type, new_token);
        list->ruby_tokens->insert({ idx, new_rtoken });
        return new_rtoken;
    } else {
        return token->second;
    }

    return Qnil;
}

VALUE llama_rb_token_list_entry(VALUE self, VALUE idx) {
    return llama_rb_token_list_entry_at(self, NUM2LONG(idx));
}

VALUE llama_rb_token_list_last(VALUE self) {
    return llama_rb_token_list_entry_at(self, -1);
}

VALUE llama_rb_token_list_length(VALUE self) {
    struct llama_rb_token_list *list;
    TypedData_Get_Struct(self, struct llama_rb_token_list, &llama_rb_token_list_type, list);

    return LONG2FIX(list->tokens->size());
}

VALUE llama_rb_token_list_each(VALUE self) {
    // @TODO figure out how to return an enumerator
    if (!rb_block_given_p()) {
        return Qnil;
    }

    struct llama_rb_token_list *list;
    TypedData_Get_Struct(self, struct llama_rb_token_list, &llama_rb_token_list_type, list);

    for (size_t i = 0; i < list->tokens->size(); i++) {
        VALUE token = llama_rb_token_list_entry_at(self, i);
        rb_yield(token);
    }

    return Qnil;
}

VALUE llama_rb_token_list_unshift(int argc, VALUE* argv, VALUE self) {
    struct llama_rb_token_list *list;
    TypedData_Get_Struct(self, struct llama_rb_token_list, &llama_rb_token_list_type, list);

    VALUE ruby_tokens;
    rb_scan_args(argc, argv, "*", &ruby_tokens);

    std::vector<llama_token> tokens((long)rb_array_len(ruby_tokens));

    for (int i = 0; i < tokens.size(); i ++) {
        tokens.at(i) = FIX2LONG(rb_ary_entry(ruby_tokens, i));
    }

    list->tokens->insert(list->tokens->begin(), tokens.begin(), tokens.end());

    // could recompute this, but meh
    list->ruby_tokens->clear();

    return Qnil;
}

VALUE llama_rb_token_list_push(VALUE self, VALUE token) {
    struct llama_rb_token_list *list;
    TypedData_Get_Struct(self, struct llama_rb_token_list, &llama_rb_token_list_type, list);

    list->tokens->push_back(FIX2LONG(token));

    return token;
}

VALUE llama_rb_token_list_clear(VALUE self) {
    struct llama_rb_token_list *list;
    TypedData_Get_Struct(self, struct llama_rb_token_list, &llama_rb_token_list_type, list);

    list->tokens->clear();

    if (list->ruby_tokens != NULL) {
        list->ruby_tokens->clear();
    }

    return Qnil;
}

VALUE llama_rb_token_to_s(VALUE self) {
    struct llama_rb_token *token;
    TypedData_Get_Struct(self, struct llama_rb_token, &llama_rb_token_type, token);

    const char *str = llama_token_to_str(token->ctx, token->token);
    return rb_str_new(str, strlen(str));
}

VALUE llama_rb_token_eos(VALUE klass) {
    return INT2FIX(llama_token_eos());
}

VALUE llama_rb_token_value(VALUE self) {
    struct llama_rb_token *token;
    TypedData_Get_Struct(self, struct llama_rb_token, &llama_rb_token_type, token);

    return LONG2FIX(token->token);
}

VALUE llama_rb_context_get_logits(VALUE self) {
    struct llama_rb_context *context;
    TypedData_Get_Struct(self, struct llama_rb_context, &llama_rb_context_type, context);

    struct llama_rb_logits *logits;
    logits = (struct llama_rb_logits *)malloc(sizeof(struct llama_rb_logits));
    logits->logits = llama_get_logits(context->ctx);
    logits->n_tokens = context->last_n_tokens;
    logits->n_vocab = llama_n_vocab(context->ctx);

    return TypedData_Wrap_Struct(llama_rb_logits_class, &llama_rb_logits_type, logits);
}

VALUE llama_rb_context_sample_top_p_top_k(VALUE self, VALUE last_n_tokens, VALUE top_k, VALUE top_p, VALUE temp, VALUE repeat_penalty) {
    struct llama_rb_context *context;
    TypedData_Get_Struct(self, struct llama_rb_context, &llama_rb_context_type, context);

    int n = rb_array_len(last_n_tokens);
    std::vector<llama_token> tokens(n);

    for (int i = 0; i < n; i ++) {
        tokens.at(i) = FIX2INT(rb_ary_entry(last_n_tokens, i));
    }

    llama_token result = llama_sample_top_p_top_k(
        context->ctx,
        tokens.data(),
        tokens.size(),
        FIX2INT(top_k),
        NUM2DBL(top_p),
        NUM2DBL(temp),
        NUM2DBL(repeat_penalty)
    );

    return INT2FIX(result);
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
void sigint_handler(int signo) {
    printf("\n"); // this also force-flushes stdout.
    if (signo == SIGINT) {
        _exit(130);
    }
}
#endif

VALUE llama_rb_context_predict_each(VALUE self, VALUE _prompt) {
    struct llama_rb_context *context;
    TypedData_Get_Struct(self, struct llama_rb_context, &llama_rb_context_type, context);

    if (context->ctx == NULL) {
        // @TODO: raise error
    }

    std::string prompt;
    prompt.assign(StringValuePtr(_prompt), RSTRING_LEN(_prompt));

    // Add a space in front of the first character to match OG llama tokenizer behavior
    prompt.insert(0, 1, ' ');

    // tokenize the prompt
    auto embd_inp = llama_tokenize(context->ctx, prompt, true);
    const int n_ctx = llama_n_ctx(context->ctx);

    // TODO: replace with ring-buffer
    std::vector<llama_token> last_n_tokens(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    // determine newline token
    auto llama_token_newline = llama_tokenize(context->ctx, "\n", false);

    gpt_params params;
    int n_predict  = 512;
    int n_keep     = params.n_keep;

    int n_past     = 0;
    int n_remain   = n_predict;
    int n_consumed = 0;

    std::vector<llama_token> embd;

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = sigint_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
    signal(SIGINT, sigint_handler);
#endif

    while (n_remain != 0) {
        // predict
        if (embd.size() > 0) {
            // infinite text generation via context swapping
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch
            if (n_past + (int) embd.size() > n_ctx) {
                const int n_left = n_past - params.n_keep;

                n_past = params.n_keep;

                // insert n_left/2 tokens at the start of embd from last_n_tokens
                embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left/2 - embd.size(), last_n_tokens.end() - embd.size());
            }

            if (llama_eval(context->ctx, embd.data(), embd.size(), n_past, params.n_threads)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return Qnil;
            }
        }

        n_past += embd.size();
        embd.clear();

        if ((int) embd_inp.size() <= n_consumed) {
            // out of user input, sample next token
            const int32_t top_k          = params.top_k;
            const float   top_p          = params.top_p;
            const float   temp           = params.temp;
            const float   repeat_penalty = params.repeat_penalty;

            llama_token id = 0;

            {
                auto logits = llama_get_logits(context->ctx);

                if (params.ignore_eos) {
                    logits[llama_token_eos()] = 0;
                }

                id = llama_sample_top_p_top_k(context->ctx,
                        last_n_tokens.data() + n_ctx - params.repeat_last_n,
                        params.repeat_last_n, top_k, top_p, temp, repeat_penalty);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);
            }

            // replace end of text token with newline token when in interactive mode
            if (id == llama_token_eos() && params.interactive && !params.instruct) {
                id = llama_token_newline.front();
                if (params.antiprompt.size() != 0) {
                    // tokenize and inject first reverse prompt
                    const auto first_antiprompt = ::llama_tokenize(context->ctx, params.antiprompt.front(), false);
                    embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                }
            }

            // add it to the context
            embd.push_back(id);

            // decrement remaining sampling budget
            --n_remain;
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[n_consumed]);
                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        // display text
        for (auto id : embd) {
            std::string str = llama_token_to_str(context->ctx, id);
            rb_yield_values(1, rb_str_new(str.c_str(), str.length()));
            // printf("%s", llama_token_to_str(context->ctx, id));
        }
        fflush(stdout);

        // end of text token
        if (embd.back() == llama_token_eos()) {
            break;
        }
    }

    return Qnil;
}

VALUE llama_rb_logits_entry(VALUE self, VALUE _row, VALUE _col) {
    struct llama_rb_logits *logits;
    TypedData_Get_Struct(self, struct llama_rb_logits, &llama_rb_logits_type, logits);

    int row = FIX2INT(_row);
    int col = FIX2INT(_col);

    if (row < 0 || row >= logits->n_tokens) return Qnil;
    if (col < 0 || col >= logits->n_vocab) return Qnil;

    size_t idx = row * logits->n_vocab + col;

    return DBL2NUM(logits->logits[idx]);
}

VALUE llama_rb_logits_set(VALUE self, VALUE _row, VALUE _col, VALUE val) {
    struct llama_rb_logits *logits;
    TypedData_Get_Struct(self, struct llama_rb_logits, &llama_rb_logits_type, logits);

    int row = FIX2INT(_row);
    int col = FIX2INT(_col);

    if (row < 0 || row >= logits->n_tokens || col < 0 || col >= logits->n_vocab) {
        rb_raise(rb_eArgError, "(%d, %d) is outside the bounds of the array", row, col);
    }

    size_t idx = row * logits->n_vocab + col;

    logits->logits[idx] = NUM2DBL(val);

    return val;
}

VALUE llama_rb_logits_length(VALUE self) {
    struct llama_rb_logits *logits;
    TypedData_Get_Struct(self, struct llama_rb_logits, &llama_rb_logits_type, logits);

    return INT2FIX(logits->n_tokens);
}

VALUE llama_rb_logits_n_cols(VALUE self) {
    struct llama_rb_logits *logits;
    TypedData_Get_Struct(self, struct llama_rb_logits, &llama_rb_logits_type, logits);

    return INT2FIX(logits->n_vocab);
}

extern "C"
void Init_llama_rb() {
    VALUE llama_rb_mod = rb_define_module("Llama");

    llama_rb_context_class = rb_define_class_under(llama_rb_mod, "Context", rb_cObject);
    rb_define_alloc_func(llama_rb_context_class, llama_rb_context_alloc);
    rb_define_method(llama_rb_context_class, "init!", RUBY_METHOD_FUNC(llama_rb_context_init), 0);
    rb_define_method(llama_rb_context_class, "tokenize", RUBY_METHOD_FUNC(llama_rb_context_tokenize), -1);
    rb_define_method(llama_rb_context_class, "n_ctx", RUBY_METHOD_FUNC(llama_rb_context_n_ctx), 0);
    rb_define_method(llama_rb_context_class, "make_token_list", RUBY_METHOD_FUNC(llama_rb_context_make_token_list), -1);
    rb_define_method(llama_rb_context_class, "evaluate", RUBY_METHOD_FUNC(llama_rb_context_evaluate), 3);
    rb_define_method(llama_rb_context_class, "predict_each", RUBY_METHOD_FUNC(llama_rb_context_predict_each), 1);
    rb_define_method(llama_rb_context_class, "logits", RUBY_METHOD_FUNC(llama_rb_context_get_logits), 0);
    rb_define_method(llama_rb_context_class, "sample_top_p_top_k", RUBY_METHOD_FUNC(llama_rb_context_sample_top_p_top_k), 5);

    llama_rb_token_class = rb_define_class_under(llama_rb_mod, "Token", rb_cObject);
    rb_define_singleton_method(llama_rb_token_class, "eos", RUBY_METHOD_FUNC(llama_rb_token_eos), 0);
    rb_define_method(llama_rb_token_class, "value", RUBY_METHOD_FUNC(llama_rb_token_value), 0);
    rb_define_method(llama_rb_token_class, "to_s", RUBY_METHOD_FUNC(llama_rb_token_to_s), 0);

    llama_rb_token_list_class = rb_define_class_under(llama_rb_mod, "TokenList", rb_cObject);
    rb_define_method(llama_rb_token_list_class, "[]", RUBY_METHOD_FUNC(llama_rb_token_list_entry), 1);
    rb_define_method(llama_rb_token_list_class, "last", RUBY_METHOD_FUNC(llama_rb_token_list_last), 0);
    rb_define_method(llama_rb_token_list_class, "size", RUBY_METHOD_FUNC(llama_rb_token_list_length), 0);
    rb_define_method(llama_rb_token_list_class, "length", RUBY_METHOD_FUNC(llama_rb_token_list_length), 0);
    rb_define_method(llama_rb_token_list_class, "each", RUBY_METHOD_FUNC(llama_rb_token_list_each), 0);
    rb_define_method(llama_rb_token_list_class, "unshift", RUBY_METHOD_FUNC(llama_rb_token_list_unshift), -1);
    rb_define_method(llama_rb_token_list_class, "<<", RUBY_METHOD_FUNC(llama_rb_token_list_push), 1);
    rb_define_method(llama_rb_token_list_class, "clear", RUBY_METHOD_FUNC(llama_rb_token_list_clear), 0);
    rb_include_module(llama_rb_token_list_class, rb_mEnumerable);

    llama_rb_logits_class = rb_define_class_under(llama_rb_mod, "Logits", rb_cObject);
    rb_define_method(llama_rb_logits_class, "[]", RUBY_METHOD_FUNC(llama_rb_logits_entry), 2);
    rb_define_method(llama_rb_logits_class, "[]=", RUBY_METHOD_FUNC(llama_rb_logits_set), 3);
    rb_define_method(llama_rb_logits_class, "size", RUBY_METHOD_FUNC(llama_rb_logits_length), 0);
    rb_define_method(llama_rb_logits_class, "n_cols", RUBY_METHOD_FUNC(llama_rb_logits_n_cols), 0);
}
