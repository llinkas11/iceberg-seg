function batch_make_locator_maps()
%BATCH_MAKE_LOCATOR_MAPS Export locator PNGs for the current example chips.
%
% This script writes locator images into:
%   /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/roboflow/locator_maps
%
% It uses inferred region prefixes from roboflow/infer_chip_region.py.
% Two chips have overlapping candidates, noted below.

rootDir = "/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026";
imgsDir = fullfile(rootDir, "S2-iceberg-areas", "S2UnetPlusPlus", "imgs");
aoiPath = fullfile(rootDir, "roboflow", "aoi_boxes.csv");
outDir = fullfile(rootDir, "roboflow", "locator_maps");

if ~isfolder(outDir)
    mkdir(outDir);
end

jobs = {
    "S2A_MSIL1C_20160725T145922_N0500_R125_T22WES_20231004T134617_pB5_9_13_",  "NK"
    "S2A_MSIL1C_20160802T141952_N0500_R096_T24WWU_20231001T172901_pB5_20_19_", "SK"
    "S2A_MSIL1C_20160802T141952_N0500_R096_T24WWU_20231001T172901_pB5_7_27_",  "SK"
    "S2A_MSIL1C_20160802T141952_N0500_R096_T24WWU_20231001T172901_pB5_8_24_",  "SK"
    "S2A_MSIL1C_20160923T145932_N0500_R125_T22WES_20231006T022513_pB5_11_14_", "NK"
    "S2A_MSIL1C_20160923T145932_N0500_R125_T22WES_20231006T022513_pB5_14_14_", "NK"
    "S2A_MSIL1C_20160923T145932_N0500_R125_T22WES_20231006T022513_pB5_16_15_", "NK"
    "S2A_MSIL1C_20160923T145932_N0500_R125_T22WES_20231006T022513_pB5_9_8_",   "NK"
    "S2A_MSIL1C_20170710T145911_N0500_R125_T22WES_20230901T062816_pB5_20_17_", "NK"
    "S2A_MSIL1C_20170814T154911_N0500_R054_T21WWU_20230830T232815_pB5_13_30_", "SQ" % also matched UQ
    "S2A_MSIL1C_20190705T141011_N0500_R053_T24WWU_20230603T191421_pB5_6_20_",  "SK"
    "S2A_MSIL1C_20190709T152911_N0500_R111_T22WDB_20230603T230701_pB5_11_36_", "II" % also matched DB
    "S2A_MSIL1C_20210731T153911_N0500_R011_T22WDB_20230224T225255_pB5_34_3_",  "DB"
    "S2B_MSIL1C_20170801T144919_N0500_R082_T22WES_20230804T124536_pB5_20_18_", "NK"
};

for k = 1:size(jobs, 1)
    stem = jobs{k, 1};
    regionPrefix = jobs{k, 2};
    chipPath = fullfile(imgsDir, stem + ".tif");
    outPath = fullfile(outDir, stem + ".png");
    fprintf("Exporting locator map %d/%d: %s (%s)\n", k, size(jobs, 1), stem, regionPrefix);
    make_locator_map(chipPath, aoiPath, outPath, regionPrefix);
end

fprintf("Locator maps written to: %s\n", outDir);
end
