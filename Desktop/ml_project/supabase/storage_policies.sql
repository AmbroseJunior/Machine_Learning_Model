-- ════════════════════════════════════════════════════════════════════════════
-- Supabase Storage Buckets and Policies
-- Run this after the main migration in the Supabase SQL editor.
-- ════════════════════════════════════════════════════════════════════════════

-- Create buckets
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES
  (
    'datasets', 'datasets', false, 104857600,  -- 100 MB
    ARRAY[
      'text/csv', 'text/tab-separated-values', 'text/plain',
      'application/json', 'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'application/octet-stream'
    ]
  ),
  (
    'models', 'models', false, 524288000,       -- 500 MB
    ARRAY['application/octet-stream']
  ),
  (
    'exports', 'exports', false, 524288000,
    ARRAY['application/octet-stream']
  )
ON CONFLICT (id) DO NOTHING;

-- datasets: users can read/write their own prefix
CREATE POLICY "datasets storage: own read"
  ON storage.objects FOR SELECT
  USING (bucket_id = 'datasets' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "datasets storage: own insert"
  ON storage.objects FOR INSERT
  WITH CHECK (bucket_id = 'datasets' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "datasets storage: service role all"
  ON storage.objects FOR ALL
  USING (bucket_id = 'datasets' AND auth.role() = 'service_role');

-- models: service role writes; authenticated users can read their own
CREATE POLICY "models storage: own read"
  ON storage.objects FOR SELECT
  USING (bucket_id = 'models' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "models storage: service role all"
  ON storage.objects FOR ALL
  USING (bucket_id = 'models' AND auth.role() = 'service_role');

-- exports: own read + service role all
CREATE POLICY "exports storage: own read"
  ON storage.objects FOR SELECT
  USING (bucket_id = 'exports' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "exports storage: service role all"
  ON storage.objects FOR ALL
  USING (bucket_id = 'exports' AND auth.role() = 'service_role');
