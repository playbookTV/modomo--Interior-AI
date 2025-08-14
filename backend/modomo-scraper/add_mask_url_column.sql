-- Add missing mask_url column to detected_objects table
-- This fixes the error: Could not find the 'mask_url' column of 'detected_objects' in the schema cache

-- Check if the column already exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'detected_objects' 
        AND column_name = 'mask_url'
    ) THEN
        -- Add the missing column
        ALTER TABLE detected_objects ADD COLUMN mask_url TEXT;
        RAISE NOTICE '✅ Added mask_url column to detected_objects table';
    ELSE
        RAISE NOTICE '⚠️  mask_url column already exists in detected_objects table';
    END IF;
END $$;

-- Verify the column was added
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'detected_objects' 
AND column_name IN ('mask_url', 'mask_r2_key')
ORDER BY column_name;