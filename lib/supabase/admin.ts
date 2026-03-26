import { createClient } from "@supabase/supabase-js";

import { getRequiredServerConfig } from "@/lib/config";

export function getSupabaseAdminClient() {
  const config = getRequiredServerConfig();

  return createClient(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY, {
    auth: {
      persistSession: false,
      autoRefreshToken: false,
    },
  });
}
