function make_locator_map(chipPath, aoiPath, outPath, regionPrefix)
%MAKE_LOCATOR_MAP Create a fjord locator map with chip outline in red.
%
% Example:
% make_locator_map( ...
%   "/Users/llinkas/.../S2-iceberg-areas/S2UnetPlusPlus/imgs/<chip>.tif", ...
%   "/Users/llinkas/.../S2-iceberg-areas/aois_greenland_area_distributions.gpkg", ...
%   "/Users/llinkas/.../roboflow/locator_map.png", ...
%   "KQ");
%
% Inputs:
%   chipPath     - path to a georeferenced chip GeoTIFF
%   aoiPath      - path to AOI boxes CSV or aois_greenland_area_distributions.gpkg
%   outPath      - output PNG path
%   regionPrefix - short fjord prefix, e.g. "KQ" or "SK"

arguments
    chipPath (1,1) string
    aoiPath (1,1) string
    outPath (1,1) string
    regionPrefix (1,1) string
end

info = georasterinfo(chipPath);
chipCRS = info.CoordinateReferenceSystem;
if ischar(chipCRS) || isstring(chipCRS)
    chipCRS = projcrs(chipCRS);
end
aoiCRS = projcrs(5938);
chipX = [info.RasterReference.XWorldLimits(1), info.RasterReference.XWorldLimits(2), ...
         info.RasterReference.XWorldLimits(2), info.RasterReference.XWorldLimits(1), ...
         info.RasterReference.XWorldLimits(1)];
chipY = [info.RasterReference.YWorldLimits(1), info.RasterReference.YWorldLimits(1), ...
         info.RasterReference.YWorldLimits(2), info.RasterReference.YWorldLimits(2), ...
         info.RasterReference.YWorldLimits(1)];

if endsWith(lower(aoiPath), ".csv")
    T = readtable(aoiPath, "TextType", "string");
    keep = startsWith(string(T.REGION), regionPrefix);
    T = T(keep, :);
    if isempty(T)
        error("No AOI boxes found for region prefix '%s'.", regionPrefix);
    end
    useBoxes = true;
else
    S = readgeotable(aoiPath);
    keep = startsWith(string(S.REGION), regionPrefix);
    S = S(keep, :);
    if isempty(S)
        error("No AOI polygons found for region prefix '%s'.", regionPrefix);
    end
    useBoxes = false;
end

chipCenterX = mean(info.RasterReference.XWorldLimits);
chipCenterY = mean(info.RasterReference.YWorldLimits);
[chipLat, chipLon] = projinv(chipCRS, chipCenterX, chipCenterY);

f = figure("Visible", "off", "Color", "white", "Position", [100 100 900 900]);
gx = geoaxes(f);
hold(gx, "on");

geobasemap(gx, "topographic");

if useBoxes
    for k = 1:height(T)
        x = [T.minx(k), T.maxx(k), T.maxx(k), T.minx(k), T.minx(k)];
        y = [T.miny(k), T.miny(k), T.maxy(k), T.maxy(k), T.miny(k)];
        [lat, lon] = projinv(aoiCRS, x, y);
        geoplot(gx, lat, lon, "Color", [0.25 0.45 0.6], "LineWidth", 1.5);
        text(gx, mean(lat(1:4)), mean(lon(1:4)), string(T.REGION(k)), ...
            "HorizontalAlignment", "center", ...
            "BackgroundColor", "white", ...
            "Margin", 2, ...
            "FontSize", 10);
    end
else
    for k = 1:height(S)
        shape = S.geom(k);
        [lat, lon] = projinv(chipCRS, shape.Vertices(:,1), shape.Vertices(:,2));
        geoplot(gx, lat, lon, "Color", [0.25 0.45 0.6], "LineWidth", 1.5);

        valid = ~isnan(lat) & ~isnan(lon);
        if any(valid)
            text(gx, mean(lat(valid)), mean(lon(valid)), string(S.REGION(k)), ...
                "HorizontalAlignment", "center", ...
                "BackgroundColor", "white", ...
                "Margin", 2, ...
                "FontSize", 10);
        end
    end
end

[chipLatPoly, chipLonPoly] = projinv(chipCRS, chipX, chipY);
geoplot(gx, chipLatPoly, chipLonPoly, "r-", "LineWidth", 2.5);
geoscatter(gx, chipLat, chipLon, 25, "r", "filled");

allLat = [];
allLon = [];
if useBoxes
    for k = 1:height(T)
        x = [T.minx(k), T.maxx(k), T.maxx(k), T.minx(k), T.minx(k)];
        y = [T.miny(k), T.miny(k), T.maxy(k), T.maxy(k), T.miny(k)];
        [lat, lon] = projinv(aoiCRS, x, y);
        allLat = [allLat; lat(:)];
        allLon = [allLon; lon(:)];
    end
else
    for k = 1:height(S)
        shape = S.geom(k);
        [lat, lon] = projinv(chipCRS, shape.Vertices(:,1), shape.Vertices(:,2));
        allLat = [allLat; lat(:)];
        allLon = [allLon; lon(:)];
    end
end

valid = ~isnan(allLat) & ~isnan(allLon);
latlim = [min(allLat(valid)) max(allLat(valid))];
lonlim = [min(allLon(valid)) max(allLon(valid))];
padLat = 0.08 * diff(latlim);
padLon = 0.08 * diff(lonlim);

if padLat == 0
    padLat = 0.05;
end
if padLon == 0
    padLon = 0.05;
end

geolimits(gx, latlim + [-padLat padLat], lonlim + [-padLon padLon]);
title(gx, sprintf("%s locator map", regionPrefix), "FontSize", 14);

subtitle(gx, sprintf("Chip center: %.4f N, %.4f W", chipLat, abs(chipLon)));

exportgraphics(f, outPath, "Resolution", 200);
close(f);
end
