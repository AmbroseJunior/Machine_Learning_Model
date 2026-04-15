-- ════════════════════════════════════════════════════════════════════════════
-- ML Studio — Initial Schema
-- ════════════════════════════════════════════════════════════════════════════

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── profiles ─────────────────────────────────────────────────────────────────
CREATE TABLE public.profiles (
  id            uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email         text NOT NULL,
  display_name  text,
  avatar_url    text,
  role          text NOT NULL DEFAULT 'user'
    CHECK (role IN ('user', 'admin')),
  api_key_hash  text,
  created_at    timestamptz NOT NULL DEFAULT now(),
  updated_at    timestamptz NOT NULL DEFAULT now()
);

-- ── datasets ─────────────────────────────────────────────────────────────────
CREATE TABLE public.datasets (
  id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id           uuid NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  name              text NOT NULL,
  original_filename text NOT NULL,
  storage_path      text NOT NULL DEFAULT '',
  file_size_bytes   bigint,
  file_format       text,
  row_count         integer,
  column_count      integer,
  column_audit      jsonb,
  status            text NOT NULL DEFAULT 'processing'
    CHECK (status IN ('processing', 'ready', 'error', 'deleted')),
  error_message     text,
  created_at        timestamptz NOT NULL DEFAULT now(),
  updated_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX datasets_user_id_idx ON public.datasets(user_id);
CREATE INDEX datasets_status_idx  ON public.datasets(status);

-- ── experiments ──────────────────────────────────────────────────────────────
CREATE TABLE public.experiments (
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id          uuid NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  dataset_id       uuid REFERENCES public.datasets(id) ON DELETE SET NULL,
  name             text NOT NULL,
  model_name       text NOT NULL,
  task             text NOT NULL DEFAULT 'unknown'
    CHECK (task IN ('classification', 'regression', 'unknown')),
  target_col       text NOT NULL,
  feature_cols     text[] NOT NULL DEFAULT '{}',
  model_params     jsonb NOT NULL DEFAULT '{}',
  pipeline_config  jsonb NOT NULL DEFAULT '{}',
  status           text NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'running', 'completed', 'failed')),
  error_message    text,
  job_id           text,
  started_at       timestamptz,
  completed_at     timestamptz,
  created_at       timestamptz NOT NULL DEFAULT now(),
  updated_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX experiments_user_id_idx  ON public.experiments(user_id);
CREATE INDEX experiments_status_idx   ON public.experiments(status);
CREATE INDEX experiments_dataset_idx  ON public.experiments(dataset_id);

-- ── experiment_metrics ───────────────────────────────────────────────────────
CREATE TABLE public.experiment_metrics (
  id                     uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  experiment_id          uuid NOT NULL REFERENCES public.experiments(id) ON DELETE CASCADE,
  accuracy               numeric(6,4),
  precision_macro        numeric(6,4),
  recall_macro           numeric(6,4),
  f1_macro               numeric(6,4),
  r2_score               numeric(8,4),
  mae                    numeric(12,4),
  rmse                   numeric(12,4),
  test_size_pct          integer,
  n_test_samples         integer,
  classification_report  jsonb,
  confusion_matrix       jsonb,
  feature_importances    jsonb,
  created_at             timestamptz NOT NULL DEFAULT now(),
  UNIQUE (experiment_id)
);

-- ── models ───────────────────────────────────────────────────────────────────
CREATE TABLE public.models (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         uuid NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  experiment_id   uuid NOT NULL REFERENCES public.experiments(id) ON DELETE CASCADE,
  name            text NOT NULL,
  version         integer NOT NULL DEFAULT 1,
  model_name      text NOT NULL,
  task            text NOT NULL,
  target_col      text NOT NULL,
  feature_cols    text[] NOT NULL DEFAULT '{}',
  col_audit       jsonb,
  col_roles       jsonb,
  storage_path    text NOT NULL,
  file_size_bytes bigint,
  is_active       boolean NOT NULL DEFAULT true,
  created_at      timestamptz NOT NULL DEFAULT now(),
  UNIQUE (user_id, name, version)
);
CREATE INDEX models_user_id_idx    ON public.models(user_id);
CREATE INDEX models_experiment_idx ON public.models(experiment_id);
CREATE INDEX models_is_active_idx  ON public.models(is_active);

-- ── predictions ──────────────────────────────────────────────────────────────
CREATE TABLE public.predictions (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     uuid NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  model_id    uuid NOT NULL REFERENCES public.models(id) ON DELETE CASCADE,
  input_data  jsonb NOT NULL,
  prediction  jsonb NOT NULL,
  latency_ms  integer,
  created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX predictions_user_id_idx    ON public.predictions(user_id);
CREATE INDEX predictions_model_id_idx   ON public.predictions(model_id);
CREATE INDEX predictions_created_at_idx ON public.predictions(created_at DESC);

-- ── audit_logs ───────────────────────────────────────────────────────────────
CREATE TABLE public.audit_logs (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       uuid REFERENCES public.profiles(id) ON DELETE SET NULL,
  action        text NOT NULL,
  resource_type text,
  resource_id   uuid,
  metadata      jsonb,
  ip_address    inet,
  user_agent    text,
  created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX audit_logs_user_id_idx    ON public.audit_logs(user_id);
CREATE INDEX audit_logs_action_idx     ON public.audit_logs(action);
CREATE INDEX audit_logs_created_at_idx ON public.audit_logs(created_at DESC);
CREATE INDEX audit_logs_resource_idx   ON public.audit_logs(resource_type, resource_id);

-- ════════════════════════════════════════════════════════════════════════════
-- Row Level Security
-- ════════════════════════════════════════════════════════════════════════════

ALTER TABLE public.profiles           ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.datasets           ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.experiments        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.experiment_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.models             ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.predictions        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.audit_logs         ENABLE ROW LEVEL SECURITY;

-- profiles
CREATE POLICY "profiles: own read"
  ON public.profiles FOR SELECT USING (auth.uid() = id);
CREATE POLICY "profiles: own update"
  ON public.profiles FOR UPDATE USING (auth.uid() = id);
CREATE POLICY "profiles: service role all"
  ON public.profiles FOR ALL USING (auth.role() = 'service_role');

-- datasets
CREATE POLICY "datasets: own all"
  ON public.datasets FOR ALL
  USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
CREATE POLICY "datasets: service role all"
  ON public.datasets FOR ALL USING (auth.role() = 'service_role');

-- experiments
CREATE POLICY "experiments: own all"
  ON public.experiments FOR ALL
  USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
CREATE POLICY "experiments: service role all"
  ON public.experiments FOR ALL USING (auth.role() = 'service_role');

-- experiment_metrics (readable if user owns parent experiment)
CREATE POLICY "experiment_metrics: own read"
  ON public.experiment_metrics FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.experiments e
      WHERE e.id = experiment_id AND e.user_id = auth.uid()
    )
  );
CREATE POLICY "experiment_metrics: service role all"
  ON public.experiment_metrics FOR ALL USING (auth.role() = 'service_role');

-- models
CREATE POLICY "models: own all"
  ON public.models FOR ALL
  USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
CREATE POLICY "models: service role all"
  ON public.models FOR ALL USING (auth.role() = 'service_role');

-- predictions
CREATE POLICY "predictions: own all"
  ON public.predictions FOR ALL
  USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
CREATE POLICY "predictions: service role all"
  ON public.predictions FOR ALL USING (auth.role() = 'service_role');

-- audit_logs (append-only for users; full access for service role)
CREATE POLICY "audit_logs: own read"
  ON public.audit_logs FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "audit_logs: service role all"
  ON public.audit_logs FOR ALL USING (auth.role() = 'service_role');

-- ════════════════════════════════════════════════════════════════════════════
-- Triggers
-- ════════════════════════════════════════════════════════════════════════════

-- Auto-create profile on new user signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
  INSERT INTO public.profiles (id, email, display_name)
  VALUES (
    new.id,
    new.email,
    COALESCE(new.raw_user_meta_data->>'display_name', split_part(new.email, '@', 1))
  );
  RETURN new;
END;
$$;

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Auto-update updated_at columns
CREATE OR REPLACE FUNCTION public.update_updated_at()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$;

CREATE TRIGGER datasets_updated_at
  BEFORE UPDATE ON public.datasets
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

CREATE TRIGGER experiments_updated_at
  BEFORE UPDATE ON public.experiments
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

CREATE TRIGGER profiles_updated_at
  BEFORE UPDATE ON public.profiles
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();
